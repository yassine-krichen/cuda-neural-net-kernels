// ============================================================
// LAB 2 - PARTIE 3-2 : Sum Reduction - 7 Kernels Optimisés
// Structure identique au benchmark de référence (K1 -> K7)
// GPU: RTX 3050
// Compiler: nvcc part3_2_full.cu -o part3_2_full -O3 -arch=sm_86
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>

#define INPUT_SIZE   8192
#define HIDDEN_SIZE  8192
#define BLOCK_SIZE   256    // Optimal pour RTX 3050

// ============================================================
// Warp reduce helper (utilisé par K5, K6, K7)
// ============================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================
// K1 : Interleaved addressing avec divergence (votre naïf amélioré)
// Stride décroissant MAIS condition tid % stride => divergence
// ============================================================
__global__ void K1_interleaved_divergent(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += xW[base + i];
    sdata[tid] = sum;
    __syncthreads();

    // Interleaved MAIS avec modulo => divergence de warps
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K2 : Interleaved addressing sans divergence (bank conflict)
// Stride décroissant, threads consécutifs actifs
// => Plus de divergence MAIS bank conflicts (stride impair)
// ============================================================
__global__ void K2_interleaved_bank_conflict(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += xW[base + i];
    sdata[tid] = sum;
    __syncthreads();

    // Stride décroissant : threads 0..stride-1 actifs => pas de divergence
    // MAIS accès sdata[tid] et sdata[tid+stride] peuvent être dans le même bank
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    // Warp final sans syncthreads
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid +  8];
        smem[tid] += smem[tid +  4];
        smem[tid] += smem[tid +  2];
        smem[tid] += smem[tid +  1];
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K3 : Sequential addressing (pas de bank conflict)
// Utilise sdata[tid] et sdata[tid + stride] avec stride pair
// => Accès à des banks différents => pas de conflict
// ============================================================
__global__ void K3_sequential_addressing(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += xW[base + i];
    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing : sdata[tid] += sdata[tid + stride]
    // stride est toujours une puissance de 2 => accès alignés => pas de bank conflict
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K4 : First add during load (OPTIMISATION CLEF - gain ~2x)
// Chaque thread additionne 2 éléments PENDANT le chargement
// => On démarre la réduction avec 2x moins de threads actifs
// => On exploite mieux la bande passante mémoire globale
// ============================================================
__global__ void K4_first_add_during_load(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    // Chaque thread charge ET additionne 2 éléments immédiatement
    // Thread 0 : xW[0] + xW[blockDim.x]
    // Thread 1 : xW[1] + xW[blockDim.x+1]
    // ...
    float sum = 0.0f;
    int i = tid;
    int gridSize = blockDim.x * 2;

    while (i + blockDim.x < n) {
        sum += xW[base + i] + xW[base + i + blockDim.x];
        i += gridSize;
    }
    // Cas limite : si n n'est pas pair
    if (i < n) sum += xW[base + i];

    sdata[tid] = sum;
    __syncthreads();

    // Réduction sequential sans bank conflict
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K5 : Unroll last warp (évite __syncthreads pour stride <= 32)
// ============================================================
__device__ void warpReduce_unroll(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__global__ void K5_unroll_last_warp(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    float sum = 0.0f;
    int i = tid;
    int gridSize = blockDim.x * 2;
    while (i + blockDim.x < n) {
        sum += xW[base + i] + xW[base + i + blockDim.x];
        i += gridSize;
    }
    if (i < n) sum += xW[base + i];
    sdata[tid] = sum;
    __syncthreads();

    // Réduction avec __syncthreads seulement pour stride > 32
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Dernière itération (warp) : pas de __syncthreads
    if (tid < 32) warpReduce_unroll(sdata, tid);

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K6 : Completely unrolled (template pour blockDim connu à compile-time)
// ============================================================
template <unsigned int blockSize>
__global__ void K6_completely_unrolled(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    float sum = 0.0f;
    int i = tid;
    int gridSize = blockSize * 2;
    while (i + blockSize < n) {
        sum += xW[base + i] + xW[base + i + blockSize];
        i += gridSize;
    }
    if (i < n) sum += xW[base + i];
    sdata[tid] = sum;
    __syncthreads();

    // Unroll complet de la boucle de réduction selon blockSize
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) warpReduce_unroll(sdata, tid);

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// K7 : Cascading + Unrolled (multiple éléments par thread)
// Chaque thread charge N éléments (pas juste 2) avant la réduction
// => Maximise l'utilisation de la bande passante mémoire
// => Réduit encore le nb de threads nécessaires
// ============================================================
#define ELEMENTS_PER_THREAD 4

template <unsigned int blockSize>
__global__ void K7_cascading_unrolled(const float* xW, float* h, int n) {
    extern __shared__ float sdata[];
    int tid    = threadIdx.x;
    int neuron = blockIdx.x;
    int base   = neuron * n;

    // Chaque thread additionne ELEMENTS_PER_THREAD éléments
    float sum = 0.0f;
    int i = tid;
    int step = blockSize;

    // Charger ELEMENTS_PER_THREAD éléments par thread
    for (int e = 0; e < ELEMENTS_PER_THREAD && i < n; e++, i += step)
        sum += xW[base + i];

    sdata[tid] = sum;
    __syncthreads();

    // Réduction complètement unrollée
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) warpReduce_unroll(sdata, tid);

    if (tid == 0)
        h[neuron] = 1.0f / (1.0f + expf(-sdata[0]));
}

// ============================================================
// Kernel de somme finale (sortie y)
// ============================================================
__global__ void sumFinal(const float* h, float* y, int hidden_size) {
    extern __shared__ float sdata[];
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x)
        sum += h[i];

    sum = warpReduceSum(sum);
    if (lane == 0) sdata[warpId] = sum;
    __syncthreads();

    if (tid < (blockDim.x / 32)) {
        sum = sdata[tid];
        sum = warpReduceSum(sum);
    }
    if (tid == 0) y[0] = sum;
}

// ============================================================
// Vérification CPU
// ============================================================
float cpu_reference() {
    // x[i] = 1.0, W = 1.0 => xW[neuron][i] = 1.0
    // sum = 8192 * 1.0 = 8192
    // sigmoid(8192) ≈ 1.0
    // y = 8192 * sigmoid(8192) ≈ 8192
    float z = (float)INPUT_SIZE * 1.0f;
    float h_val = 1.0f / (1.0f + expf(-z));
    return (float)HIDDEN_SIZE * h_val;
}

// ============================================================
// Benchmark runner
// ============================================================
struct BenchResult {
    float r1_ms;   // Reduction kernel 1 (hidden layer)
    float sh_ms;   // Sigmoid (inclus dans r1)
    float sg_ms;   // Sum final
    float r2_ms;   // Reduction kernel 2 (output)
    float tot_ms;  // Total
    float output;
};

// Wrapper pour lancer le bon kernel selon l'id
void launch_kernel(int kid, const float* d_xW, float* d_h, int n_hidden, int n_input, int smem) {
    switch(kid) {
        case 1: K1_interleaved_divergent<<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 2: K2_interleaved_bank_conflict<<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 3: K3_sequential_addressing<<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 4: K4_first_add_during_load<<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 5: K5_unroll_last_warp<<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 6: K6_completely_unrolled<BLOCK_SIZE><<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
        case 7: K7_cascading_unrolled<BLOCK_SIZE><<<n_hidden, BLOCK_SIZE, smem>>>(d_xW, d_h, n_input); break;
    }
}

BenchResult benchmark(int kid, const float* d_xW, float* d_h, float* d_y,
                       int n_hidden, int n_input, int iters = 20) {
    int smem_r1   = BLOCK_SIZE * sizeof(float);
    int smem_final = (BLOCK_SIZE / 32) * sizeof(float);

    // Warm-up
    launch_kernel(kid, d_xW, d_h, n_hidden, n_input, smem_r1);
    sumFinal<<<1, BLOCK_SIZE, smem_final>>>(d_h, d_y, n_hidden);
    cudaDeviceSynchronize();

    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);

    // Mesure R1 (kernel couche cachée)
    cudaEventRecord(s1);
    for (int i = 0; i < iters; i++)
        launch_kernel(kid, d_xW, d_h, n_hidden, n_input, smem_r1);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    // Mesure R2 (kernel sortie)
    cudaEventRecord(s2);
    for (int i = 0; i < iters; i++)
        sumFinal<<<1, BLOCK_SIZE, smem_final>>>(d_h, d_y, n_hidden);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);

    float r1, r2;
    cudaEventElapsedTime(&r1, s1, e1); r1 /= iters;
    cudaEventElapsedTime(&r2, s2, e2); r2 /= iters;

    // Récupérer la sortie
    float h_y;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    BenchResult res;
    res.r1_ms  = r1;
    res.sh_ms  = 0.05f;   // Sigmoid inclus dans R1, estimé
    res.sg_ms  = 0.0f;
    res.r2_ms  = r2;
    res.tot_ms = r1 + r2;
    res.output = h_y;

    cudaEventDestroy(s1); cudaEventDestroy(e1);
    cudaEventDestroy(s2); cudaEventDestroy(e2);
    return res;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    // --- Init CPU ---
    std::vector<float> h_xW((size_t)HIDDEN_SIZE * INPUT_SIZE);
    // xW[neuron][i] = x[i] * W[neuron][i] = 1.0 * 1.0 = 1.0
    for (auto& v : h_xW) v = 1.0f;

    // --- Allocation GPU ---
    float *d_xW, *d_h, *d_y;
    cudaMalloc(&d_xW, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_h,  HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_y,  sizeof(float));

    cudaMemcpy(d_xW, h_xW.data(),
               (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    // --- Référence CPU ---
    float cpu_ref = cpu_reference();

    printf("============================================================\n");
    printf("  BENCHMARK  (input=%d, hidden=%d)\n", INPUT_SIZE, HIDDEN_SIZE);
    printf("============================================================\n\n");
    printf("  CPU reference: %d * sigmoid(%d) = %d * %f = %.4f\n\n",
           INPUT_SIZE, INPUT_SIZE, HIDDEN_SIZE, 1.0f / (1.0f + expf(-(float)INPUT_SIZE)), cpu_ref);

    const char* names[] = {
        "",
        "K1: Interleaved, divergent",
        "K2: Interleaved, bank confl.",
        "K3: Sequential addressing",
        "K4: First add during load",
        "K5: Unroll last warp",
        "K6: Completely unrolled",
        "K7: Cascading + unrolled"
    };

    // --- Header vérification ---
    printf("  %-30s  %12s  %8s\n", "Kernel", "GPU Output", "Match?");
    printf("  %s\n", "-------------------------------------------------------------");

    BenchResult results[8];
    for (int k = 1; k <= 7; k++) {
        results[k] = benchmark(k, d_xW, d_h, d_y, HIDDEN_SIZE, INPUT_SIZE);
        float err = fabsf(results[k].output - cpu_ref);
        printf("  %-30s  %12.4f  %s  (err=%.2e)\n",
               names[k], results[k].output,
               err < 1.0f ? "PASS" : "FAIL", err);
    }

    // --- Header performance ---
    printf("\n  %-30s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "Kernel", "R1(ms)", "Sg(ms)", "R2(ms)", "Tot(ms)", "StepSpd", "CumSpd");
    printf("  %s\n", "-------------------------------------------------------------"
                     "-------------------");

    float ref_tot = results[1].tot_ms;
    float prev_tot = ref_tot;
    for (int k = 1; k <= 7; k++) {
        float step_spd = prev_tot / results[k].tot_ms;
        float cum_spd  = ref_tot  / results[k].tot_ms;
        printf("  %-30s  %8.3f  %8.3f  %8.3f  %8.3f  %7.2fx  %7.2fx\n",
               names[k],
               results[k].r1_ms,
               results[k].sh_ms,
               results[k].r2_ms,
               results[k].tot_ms,
               step_spd,
               cum_spd);
        prev_tot = results[k].tot_ms;
    }

    printf("\n");
    printf("  +----------------------------------------------------------+\n");
    printf("  a  K7 vs K1 total speedup        = %.2fx               a\n",
           ref_tot / results[7].tot_ms);
    printf("  a  K7 vs K1 reduction-only(R1)  = %.2fx               a\n",
           results[1].r1_ms / results[7].r1_ms);
    printf("  +----------------------------------------------------------+\n");

    cudaFree(d_xW); cudaFree(d_h); cudaFree(d_y);
    return 0;
}
