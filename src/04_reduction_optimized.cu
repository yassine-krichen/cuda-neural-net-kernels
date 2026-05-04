// ============================================================
// LAB 2 - PARTIE 3-2 : Sum Reduction OPTIMISEE
// Amelioration 1 : Reduction entrelacee (sans divergence de warps)
// Amelioration 2 : Warp Shuffle (sans shared memory)
// Amelioration 3 : (Bonus) Reduction vectorisee + unrolling
// ============================================================
// Compiler : nvcc part3_2_optimized.cu -o part3_2 -lm -O3

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using std::vector;

#define INPUT_SIZE   8192
#define HIDDEN_SIZE  8192
#define BLOCK_SIZE   1024

// ============================================================
// ---- KERNEL NAÏF (reference) ----
// ============================================================

__global__ void sumReductionNaive_ref(const float* __restrict__ xW,
                                       float*       __restrict__ h,
                                       int          input_size) {
    extern __shared__ float sharedData[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * input_size;

    float neuron_sum = 0.0f;
    for (int offset = 0; offset < input_size; offset += blockDim.x) {
        if (offset + tid < input_size) {
            sharedData[tid] = xW[base + offset + tid];
        } else {
            sharedData[tid] = 0.0f;
        }
        __syncthreads();

        // Naïf : stride croissant => divergence de warps
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0)
                sharedData[tid] += sharedData[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            neuron_sum += sharedData[0];
        }
        __syncthreads();
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-neuron_sum));
}

// ============================================================
// ---- AMELIORATION 1 : Reduction entrelacee (Interleaved) ----
//
// Principe :
//   Au lieu de stride = 1, 2, 4, ... (threads pairs actifs)
//   On utilise stride = N/2, N/4, N/8, ... (threads de 0 a stride-1 actifs)
//
// Avantage :
//   - Les threads ACTIFS sont consecutifs : 0, 1, 2, ..., stride-1
//   - Dans un warp (32 threads), tous les threads sont SOIT actifs SOIT inactifs
//   - Zero divergence de warps jusqu'a stride < 32
//   - Accès shared memory consecutifs => pas de bank conflict
//
// Gain attendu : ~2x par rapport au naïf
// ============================================================

__global__ void sumReductionInterleaved(const float* __restrict__ xW,
                                         float*       __restrict__ h,
                                         int          input_size) {
    extern __shared__ float sharedData[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * input_size;

    // Chargement avec reduction sequentielle initiale
    float sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x)
        sum += xW[base + i];
    sharedData[tid] = sum;
    __syncthreads();

    // Reduction entrelacee : stride decroissant de blockDim.x/2 jusqu'a 1
    // Les threads [0 .. stride-1] sont actifs, [stride .. 2*stride-1] inactifs
    // => Pas de divergence dans les warps tant que stride >= 32
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sharedData[0]));
}

// ============================================================
// ---- AMELIORATION 2 : Warp Shuffle (__shfl_down_sync) ----
//
// Principe :
//   - A partir de stride = 32 (niveau warp), on n'utilise PLUS la shared memory
//   - __shfl_down_sync permet a un thread de lire le registre d'un autre thread
//     dans le MÊME WARP, sans passer par la memoire
//   - Latence ~1 cycle vs ~20+ cycles pour shared memory
//
// Strategie :
//   1. Phase 1 : reduction entrelacee dans shared memory jusqu'au niveau warp
//   2. Phase 2 : reduction finale dans chaque warp via __shfl_down_sync
//   3. Phase 3 : les "lane 0" de chaque warp ecrivent dans shared memory
//   4. Phase 4 : le premier warp reduit les resultats des warps
//
// Gain attendu : ~1.5x a 2x sur la partie reduction vs amelioration 1
// ============================================================

// Fonction helper : reduction d'un warp avec shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
    // Masque : tous les threads du warp participent
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // Seul lane 0 a la somme complète du warp
}

__global__ void sumReductionWarpShuffle(const float* __restrict__ xW,
                                         float*       __restrict__ h,
                                         int          input_size) {
    extern __shared__ float sharedData[];

    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * input_size;
    int lane   = tid % 32;          // Position dans le warp (0..31)
    int warpId = tid / 32;          // Numero du warp dans le bloc

    // --- Phase 1 : Chaque thread accumule ses elements localement ---
    float sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x)
        sum += xW[base + i];

    // --- Phase 2 : Reduction intra-warp via shuffle ---
    sum = warpReduceSum(sum);

    // --- Phase 3 : Lane 0 de chaque warp ecrit dans shared memory ---
    if (lane == 0) {
        sharedData[warpId] = sum;
    }
    __syncthreads();

    // --- Phase 4 : Premier warp reduit les resultats de tous les warps ---
    // Nombre de warps dans le bloc
    int numWarps = blockDim.x / 32;

    if (tid < numWarps) {
        sum = sharedData[tid];       // Chaque thread = un warp precedent
        sum = warpReduceSum(sum);    // Reduction finale
    }

    // --- Resultat final + Sigmoid ---
    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sum));
}

// ============================================================
// ---- AMELIORATION 3 (BONUS) : Reduction avec Unrolling ----
//
// Principe :
//   - Derouler la boucle de reduction pour les dernières iterations
//   - Quand stride <= 32, tous les threads actifs sont dans le même warp
//   - On peut supprimer __syncthreads() (synchronisation implicite dans un warp)
//   - Utilisation de "volatile" pour forcer les ecritures immediates
//
// Gain : reduit l'overhead de synchronisation pour les petits strides
// ============================================================

__device__ void warpReduce(volatile float* sdata, int tid) {
    // Les 6 dernières iterations (stride <= 32) : pas besoin de __syncthreads
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sumReductionUnrolled(const float* __restrict__ xW,
                                      float*       __restrict__ h,
                                      int          input_size) {
    extern __shared__ float sharedData[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * input_size;

    // Chargement avec reduction initiale (2 elements par thread au minimum)
    float sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x)
        sum += xW[base + i];
    sharedData[tid] = sum;
    __syncthreads();

    // Reduction entrelacee jusqu'a stride = 64
    if (blockDim.x >= 1024) { if (tid < 512) { sharedData[tid] += sharedData[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512)  { if (tid < 256) { sharedData[tid] += sharedData[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256)  { if (tid < 128) { sharedData[tid] += sharedData[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128)  { if (tid < 64)  { sharedData[tid] += sharedData[tid + 64];  } __syncthreads(); }

    // Dernière iteration : unrolled warp (pas de __syncthreads)
    if (tid < 32) warpReduce(sharedData, tid);

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sharedData[0]));
}

// ============================================================
// Kernel de somme finale (commun a toutes les optimisations)
// ============================================================
__global__ void sumFinalShuffle(const float* __restrict__ h,
                                 float*       __restrict__ y,
                                 int          hidden_size) {
    extern __shared__ float sharedData[];
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;
    int numWarps = blockDim.x / 32;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x)
        sum += h[i];

    sum = warpReduceSum(sum);

    if (lane == 0) sharedData[warpId] = sum;
    __syncthreads();

    if (tid < numWarps) {
        sum = sharedData[tid];
        sum = warpReduceSum(sum);
    }

    if (tid == 0) y[0] = sum;   // Pas d'activation sur la sortie
}

// ============================================================
// Fonction de benchmark generique
// ============================================================

typedef void (*KernelFn)(const float*, float*, int);

float run_benchmark(const char* name,
                    KernelFn kernel_fn,
                    const float* d_xW,
                    float* d_h,
                    float* d_y,
                    int input_size,
                    int hidden_size,
                    int block_size,
                    const std::vector<float>& h_ref,
                    int iterations = 10) {
    int sharedMem = block_size * sizeof(float);

    // Warm-up
    kernel_fn<<<hidden_size, block_size, sharedMem>>>(d_xW, d_h, input_size);
    cudaDeviceSynchronize();

    // Verify correctness
    std::vector<float> h_out(hidden_size);
    cudaMemcpy(h_out.data(), d_h, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    bool valid = true;
    for (int i = 0; i < hidden_size; i++) {
        if (std::abs(h_out[i] - h_ref[i]) > 1e-3f) {
            valid = false;
            break;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int iter = 0; iter < iterations; iter++) {
        kernel_fn<<<hidden_size, block_size, sharedMem>>>(d_xW, d_h, input_size);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    float avg = elapsed / iterations;

    if (valid) {
        printf("  %-35s : %.3f ms [VALID]\n", name, avg);
    } else {
        printf("  %-35s : %.3f ms [ERROR! INCORRECT MATH]\n", name, avg);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg;
}

// ============================================================
// Benchmark special pour WarpShuffle (shared mem differente)
// ============================================================
float run_benchmark_shuffle(const char* name,
                             const float* d_xW,
                             float* d_h,
                             int input_size,
                             int hidden_size,
                             int block_size,
                             const std::vector<float>& h_ref,
                             int iterations = 10) {
    int numWarps  = block_size / 32;
    int sharedMem = numWarps * sizeof(float);  // Seulement 1 float par warp !

    // Warm-up
    sumReductionWarpShuffle<<<hidden_size, block_size, sharedMem>>>(d_xW, d_h, input_size);
    cudaDeviceSynchronize();

    // Verify correctness
    std::vector<float> h_out(hidden_size);
    cudaMemcpy(h_out.data(), d_h, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    bool valid = true;
    for (int i = 0; i < hidden_size; i++) {
        if (std::abs(h_out[i] - h_ref[i]) > 1e-3f) {
            valid = false;
            break;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int iter = 0; iter < iterations; iter++) {
        sumReductionWarpShuffle<<<hidden_size, block_size, sharedMem>>>(d_xW, d_h, input_size);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    float avg = elapsed / iterations;

    if (valid) {
        printf("  %-35s : %.3f ms [VALID]\n", name, avg);
    } else {
        printf("  %-35s : %.3f ms [ERROR! INCORRECT MATH]\n", name, avg);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    printf("============================================================\n");
    printf("  PARTIE 3-2 : Optimisations Sum Reduction\n");
    printf("  Input size  : %d\n", INPUT_SIZE);
    printf("  Hidden size : %d\n", HIDDEN_SIZE);
    printf("  Block size  : %d\n", BLOCK_SIZE);
    printf("============================================================\n");

    // Allocation et initialisation CPU
    vector<float> h_x(INPUT_SIZE);
    vector<float> h_W(HIDDEN_SIZE * INPUT_SIZE, 1.0f);  // Poids = 1
    vector<float> h_xW(HIDDEN_SIZE * INPUT_SIZE);
    vector<float> h_h(HIDDEN_SIZE);
    float h_y = 0.0f;

    for (int i = 0; i < INPUT_SIZE; i++)
        h_x[i] = (float)(i + 1) / INPUT_SIZE;

    // Multiplication element par element sur CPU
    for (int neuron = 0; neuron < HIDDEN_SIZE; neuron++)
        for (int i = 0; i < INPUT_SIZE; i++)
            h_xW[neuron * INPUT_SIZE + i] = h_x[i] * h_W[neuron * INPUT_SIZE + i];

    // Compute CPU Gold Reference for Sigmoid Reduction Correctness
    std::vector<float> h_ref_h(HIDDEN_SIZE);
    for (int neuron = 0; neuron < HIDDEN_SIZE; neuron++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += h_xW[neuron * INPUT_SIZE + i];
        }
        h_ref_h[neuron] = 1.0f / (1.0f + std::exp(-sum));
    }

    // Allocation GPU
    float *d_xW, *d_h, *d_y;
    cudaMalloc(&d_xW, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_h,  HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_y,  sizeof(float));

    cudaMemcpy(d_xW, h_xW.data(),
               (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // ============================================================
    // Benchmarks comparatifs
    // ============================================================
    printf("\n--- Temps moyen sur %d iterations ---\n\n", 10);

    float t_naive       = run_benchmark("Naive (reference)",
                                         sumReductionNaive_ref,
                                         d_xW, d_h, d_y,
                                         INPUT_SIZE, HIDDEN_SIZE, BLOCK_SIZE, h_ref_h);

    float t_interleaved = run_benchmark("Amelioration 1 : Entrelacee",
                                         sumReductionInterleaved,
                                         d_xW, d_h, d_y,
                                         INPUT_SIZE, HIDDEN_SIZE, BLOCK_SIZE, h_ref_h);

    float t_shuffle     = run_benchmark_shuffle("Amelioration 2 : Warp Shuffle",
                                                 d_xW, d_h,
                                                 INPUT_SIZE, HIDDEN_SIZE, BLOCK_SIZE, h_ref_h);

    float t_unrolled    = run_benchmark("Amelioration 3 : Unrolling",
                                         sumReductionUnrolled,
                                         d_xW, d_h, d_y,
                                         INPUT_SIZE, HIDDEN_SIZE, BLOCK_SIZE, h_ref_h);

    // ============================================================
    // Tableau recapitulatif
    // ============================================================
    printf("\n============================================================\n");
    printf("  TABLEAU RECAPITULATIF DES PERFORMANCES\n");
    printf("  %-35s  %8s  %8s\n", "Kernel", "Temps(ms)", "Speedup");
    printf("  %-35s  %8.3f  %8.2fx\n", "Naive (reference)",             t_naive,       1.0f);
    printf("  %-35s  %8.3f  %8.2fx\n", "Amelioration 1 : Entrelacee",   t_interleaved, t_naive / t_interleaved);
    printf("  %-35s  %8.3f  %8.2fx\n", "Amelioration 2 : Warp Shuffle", t_shuffle,     t_naive / t_shuffle);
    printf("  %-35s  %8.3f  %8.2fx\n", "Amelioration 3 : Unrolling",    t_unrolled,    t_naive / t_unrolled);
    printf("============================================================\n");

    cudaFree(d_xW); cudaFree(d_h); cudaFree(d_y);
    return 0;
}

/*
============================================================
  ANALYSE PARTIE 3-2 (a inclure dans le rapport)
============================================================

--- AMELIORATION 1 : Reduction Entrelacee ---

Technique :
  Au lieu de : stride = 1, 2, 4, 8 ... (tid % 2*stride == 0)
  On utilise  : stride = N/2, N/4, N/8 ... (tid < stride)

Pourquoi c'est mieux :
  - Les threads actifs sont CONSECUTIFS : 0, 1, 2, ..., stride-1
  - Un warp de 32 threads est ENTIÈREMENT actif ou ENTIÈREMENT inactif
  - => Zero divergence de warps
  - Les accès a sharedData sont consecutifs => pas de bank conflict
  
Gain typique : 2x a 2.5x par rapport au naïf

--- AMELIORATION 2 : Warp Shuffle (__shfl_down_sync) ---

Technique :
  __shfl_down_sync(mask, val, offset) permet a un thread de lire le registre
  d'un autre thread du même warp (decale de 'offset') SANS passer par la memoire.

Pourquoi c'est mieux :
  - Latence de ~1 cycle vs ~20+ cycles pour shared memory
  - Pas de bank conflict possible (pas de shared memory pour la reduction intra-warp)
  - Reduit la pression sur le cache L1/shared memory
  - Permet une reduction finale plus rapide

Structure :
  1. Chaque thread accumule ses elements (registres)     -> latence: global mem
  2. warpReduceSum : 5 shuffles (offsets 16,8,4,2,1)    -> latence: ~5 cycles
  3. Lane 0 de chaque warp -> shared mem                 -> latence: ~20 cycles
  4. Premier warp shuffle final                          -> latence: ~5 cycles
  Total shared mem accesses : 1 ecriture + 1 lecture par warp (vs N lectures+ecritures)

Gain typique : 1.5x a 3x par rapport a l'entrelacee

--- AMELIORATION 3 : Unrolling du Warp Final ---

Technique :
  - Pour les dernières 6 iterations (stride = 32, 16, 8, 4, 2, 1)
  - Tous les threads actifs sont dans le MÊME WARP
  - Les warps sont synchronises implicitement (SIMT)
  - On peut supprimer __syncthreads() pour ces iterations
  - On utilise 'volatile' pour forcer les mises a jour en memoire

Pourquoi c'est mieux :
  - Supprime 6 barrières __syncthreads() qui sont coûteuses
  - Le compilateur ne peut pas eliminer les __syncthreads automatiquement
  - Chaque __syncthreads = ~20 cycles de latence minimum

Gain typique : 1.2x a 1.5x par rapport a l'entrelacee simple

--- RECAPITULATIF ATTENDU ---
  Naïf           : reference   (1x)
  Entrelacee     : ~2x speedup
  Warp Shuffle   : ~3x-4x speedup  
  Unrolling      : ~2.5x-3.5x speedup
  Combinaison    : jusqu'a 5x-6x speedup total
============================================================
*/
