// ============================================================
// LAB 2 - PARTIE 3-1 : Sum Reduction Naïve + Réseau de neurones
// Single hidden layer neural network (8192 entrées, 8192 neurones)
// ============================================================
// Compiler : nvcc part3_1_naive.cu -o part3_1 -lm

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using std::vector;
using std::cout;

// ============================================================
// Constantes
// ============================================================
#define INPUT_SIZE   8192
#define HIDDEN_SIZE  8192

// Taille de bloc : 8192 éléments à sommer par neurone
// => on utilise 1024 threads (max autorisé par CUDA)
// => chaque thread charge 8192/1024 = 8 éléments (réduction séquentielle en entrée)
// Voir explication dans les commentaires en bas
#define BLOCK_SIZE   1024

// ============================================================
// STEP 1 (CPU) : Multiplication élément par élément
// Produit x[i] * W[neuron][i] pour tous les neurones
// ============================================================

void elementwise_multiply_cpu(const float* x,
                               const float* W,
                               float*       xW,
                               int input_size,
                               int hidden_size) {
    // xW[neuron * input_size + i] = x[i] * W[neuron * input_size + i]
    for (int neuron = 0; neuron < hidden_size; neuron++) {
        for (int i = 0; i < input_size; i++) {
            xW[neuron * input_size + i] = x[i] * W[neuron * input_size + i];
        }
    }
}

// ============================================================
// STEP 2 (GPU) : Sum Reduction NAÏVE + Sigmoid
//
// Chaque bloc traite UN neurone (une ligne de xW).
// Les threads du bloc somment les input_size valeurs de ce neurone.
//
// Algorithme naïf (voir Table-1 du sujet) :
//   - stride = 1, 2, 4, 8, ... jusqu'à blockDim.x
//   - Seuls les threads avec tid % (2*stride) == 0 sont actifs
//   => Divergence de warps ! (inefficace, mais simple)
//
// Activation : sigmoid(x) = 1 / (1 + e^(-x))
// ============================================================

__global__ void sumReductionNaiveNN(const float* __restrict__ xW,
                                     float*       __restrict__ h,
                                     int          input_size) {
    // Shared memory : chaque bloc charge input_size éléments
    // Mais on ne peut pas dépasser 48 KB => on limite à BLOCK_SIZE
    extern __shared__ float sharedData[];

    int tid    = threadIdx.x;
    int neuron = blockIdx.x;  // Un bloc = un neurone
    int base   = neuron * input_size;

    float neuron_sum = 0.0f;
    for (int offset = 0; offset < input_size; offset += blockDim.x) {
        if (offset + tid < input_size) {
            sharedData[tid] = xW[base + offset + tid];
        } else {
            sharedData[tid] = 0.0f;
        }
        __syncthreads();

        // Naïve reduction within block
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0) {
                sharedData[tid] += sharedData[tid + stride];
            }
            __syncthreads(); // Synchronize after each step
        }

        if (tid == 0) {
            neuron_sum += sharedData[0];
        }
        __syncthreads();
    }

    // Write block result to output
    if (tid == 0) {
        h[neuron] = 1.0f / (1.0f + expf(-neuron_sum));
    }
}

// ============================================================
// STEP 3 (GPU) : Somme finale pour y = sum(h[i])
// (sortie layer : 1 seul neurone, sans activation)
// ============================================================

__global__ void sumReductionNaiveFinal(const float* __restrict__ h,
                                        float*       __restrict__ y,
                                        int          hidden_size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;

    float neuron_sum = 0.0f;
    for (int offset = 0; offset < hidden_size; offset += blockDim.x) {
        if (offset + tid < hidden_size) {
            sharedData[tid] = h[offset + tid];
        } else {
            sharedData[tid] = 0.0f;
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0) {
                sharedData[tid] += sharedData[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            neuron_sum += sharedData[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[0] = neuron_sum;   // Pas d'activation sur la sortie
    }
}

// ============================================================
// MAIN
// ============================================================

int main() {
    printf("============================================================\n");
    printf("  PARTIE 3-1 : Réseau de neurones - Sum Reduction Naïve\n");
    printf("  Input size  : %d\n", INPUT_SIZE);
    printf("  Hidden size : %d\n", HIDDEN_SIZE);
    printf("  Block size  : %d\n", BLOCK_SIZE);
    printf("============================================================\n");

    // --- Allocation et initialisation CPU ---
    // x : vecteur d'entrée (8192 éléments)
    // W : matrice des poids (8192 x 8192), tous = 1.0 (selon le sujet)
    // xW : résultat de la multiplication élément par élément (sur CPU)
    // h : sorties de la couche cachée (8192 neurones)
    // y : sortie finale (1 neurone)

    vector<float> h_x(INPUT_SIZE);
    vector<float> h_W(HIDDEN_SIZE * INPUT_SIZE);
    vector<float> h_xW(HIDDEN_SIZE * INPUT_SIZE);
    vector<float> h_h(HIDDEN_SIZE);
    float         h_y = 0.0f;

    // Initialisation : x[i] = i+1 (valeurs de test), W[i][j] = 1.0
    for (int i = 0; i < INPUT_SIZE; i++) {
        h_x[i] = (float)(i + 1) / INPUT_SIZE;  // valeurs normalisées
    }
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W[i] = 1.0f;  // Tous les poids = 1 (comme indiqué dans le sujet)
    }

    // ---- STEP 1 : Multiplication élément par élément (CPU) ----
    printf("\n[STEP 1] Multiplication element-par-element sur CPU...\n");
    elementwise_multiply_cpu(h_x.data(), h_W.data(), h_xW.data(),
                              INPUT_SIZE, HIDDEN_SIZE);

    // ---- Allocation GPU ----
    float *d_xW, *d_h, *d_y;
    size_t bytes_xW = (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(float);
    size_t bytes_h  = HIDDEN_SIZE * sizeof(float);
    size_t bytes_y  = sizeof(float);

    cudaMalloc(&d_xW, bytes_xW);
    cudaMalloc(&d_h,  bytes_h);
    cudaMalloc(&d_y,  bytes_y);

    // ---- Mesure du temps GPU ----
    cudaEvent_t start, h2d_done, kernel1_done, kernel2_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&kernel1_done);
    cudaEventCreate(&kernel2_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // ---- Transfert Host -> Device ----
    printf("[STEP 2] Transfert xW vers GPU (%zu Mo)...\n", bytes_xW / (1024*1024));
    cudaMemcpy(d_xW, h_xW.data(), bytes_xW, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_done, 0);

    // ---- STEP 2 : Sum Reduction + Sigmoid (GPU) ----
    // Un bloc par neurone => HIDDEN_SIZE blocs
    // BLOCK_SIZE threads par bloc
    // Shared memory = BLOCK_SIZE * sizeof(float) par bloc
    printf("[STEP 3] Sum Reduction Naive + Sigmoid sur GPU...\n");
    printf("         Grid : %d blocs x %d threads\n", HIDDEN_SIZE, BLOCK_SIZE);

    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    sumReductionNaiveNN<<<HIDDEN_SIZE, BLOCK_SIZE, sharedMemSize>>>(
        d_xW, d_h, INPUT_SIZE
    );
    cudaEventRecord(kernel1_done, 0);

    // ---- STEP 3 : Somme finale pour y ----
    sumReductionNaiveFinal<<<1, BLOCK_SIZE, sharedMemSize>>>(
        d_h, d_y, HIDDEN_SIZE
    );
    cudaEventRecord(kernel2_done, 0);

    // ---- Transfert Device -> Host ----
    cudaMemcpy(h_h.data(), d_h, bytes_h, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_y,        d_y, bytes_y, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // ---- Calcul des temps ----
    float t_h2d, t_k1, t_k2, t_d2h, t_total;
    cudaEventElapsedTime(&t_h2d,  start,        h2d_done);
    cudaEventElapsedTime(&t_k1,   h2d_done,     kernel1_done);
    cudaEventElapsedTime(&t_k2,   kernel1_done, kernel2_done);
    cudaEventElapsedTime(&t_d2h,  kernel2_done, stop);
    cudaEventElapsedTime(&t_total,start,         stop);

    printf("\n=== RESULTATS ===\n");
    printf("  Sortie y = %.6f\n", h_y);
    printf("  h[0]     = %.6f\n", h_h[0]);
    printf("  h[8191]  = %.6f\n", h_h[HIDDEN_SIZE-1]);

    printf("\n=== PERFORMANCES (GPU) ===\n");
    printf("  Host->Device (xW) : %.3f ms\n", t_h2d);
    printf("  Kernel 1 (Sum+Sig): %.3f ms\n", t_k1);
    printf("  Kernel 2 (Final)  : %.3f ms\n", t_k2);
    printf("  Device->Host      : %.3f ms\n", t_d2h);
    printf("  TOTAL GPU         : %.3f ms\n", t_total);

    // Libération
    cudaFree(d_xW); cudaFree(d_h); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(h2d_done);
    cudaEventDestroy(kernel1_done); cudaEventDestroy(kernel2_done);
    cudaEventDestroy(stop);

    return 0;
}

/*
============================================================
  ANALYSE PARTIE 3-1 (à inclure dans le rapport)
============================================================

--- 3-1-A : Inefficacités du kernel naïf ---

1. DIVERGENCE DE WARPS (problème principal) :
   - La condition `tid % (2*stride) == 0` crée des branches dans le warp
   - Ex: à stride=1, threads 0,2,4,6... actifs, threads 1,3,5... inactifs
   - Les threads d'un même warp (32 threads) exécutent des chemins différents
   - Le hardware DOIT exécuter les 2 chemins séquentiellement => 2x plus lent
   - À chaque itération, la moitié des threads est idle => utilisation GPU ~50%

2. CONFLITS DE BANQUE EN SHARED MEMORY :
   - À stride=1 : thread 0 accède à sharedData[0] et sharedData[1]
   - Ces accès peuvent tomber dans le même bank => serialisation
   - 32 banks dans la shared memory, adresses multiples de 4 bytes

3. THREADS INACTIFS NOMBREUX :
   - Au bout de log2(N) itérations, seul 1 thread sur N est actif
   - Pour N=1024 : après 10 itérations, seul thread 0 travaille
   - 99.9% des threads sont inactifs mais le warp reste actif

4. MAUVAISE UTILISATION DES WARPS :
   - Les threads inactifs occupent toujours des slots de warp
   - L'occupancy effective diminue rapidement

--- 3-1-B : Choix du Block Size ---

Choix : BLOCK_SIZE = 1024

Justification :
  - input_size = 8192 éléments par neurone
  - Avec 1024 threads, chaque thread charge 8192/1024 = 8 éléments
    (boucle de chargement dans le kernel, additionne localement avant la réduction)
  - 1024 = maximum autorisé par CUDA (compute capability >= 2.0)
  - Maximise l'utilisation du SM
  - Shared memory utilisée : 1024 * 4 = 4 Ko (bien dans les 48 Ko disponibles)
  - Nombre de blocs = 8192 (un par neurone) => bonne occupation du GPU

Alternatives :
  - 256 threads : nécessite 32 éléments/thread, plus de calcul séquentiel
  - 512 threads : compromis acceptable

--- 3-1-C : Appel kernel dans le Host ---

// Kernel 1 : reduction + sigmoid pour chaque neurone
sumReductionNaiveNN<<<HIDDEN_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(
    d_xW, d_h, INPUT_SIZE
);

// Kernel 2 : somme finale des h[i] pour y
sumReductionNaiveFinal<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(
    d_h, d_y, HIDDEN_SIZE
);

Les 2 kernels sont lancés séquentiellement (pas besoin de synchronisation
explicite car ils sont dans le même stream par défaut).
============================================================
*/
