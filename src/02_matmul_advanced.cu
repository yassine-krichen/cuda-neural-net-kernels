// ============================================================
// LAB 2 - PARTIE 2 : cuBLAS & Tensor Cores
// Multiplication de matrices N=8192
// Compiler : nvcc part2_cublas_tensor.cu -o part2 -lcublas
// ============================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

// cuBLAS
#include <cublas_v2.h>

// Tensor Cores via WMMA (Warp Matrix Multiply Accumulate)
// Nécessite GPU Volta (V100) ou plus récent (Turing, Ampere)
#include <mma.h>

using namespace nvcuda;
using std::vector;
using std::cout;
using std::generate;

// ============================================================
// PARTIE 2-1 : cuBLAS
// ============================================================

/*
  cuBLAS utilise la convention COLUMN-MAJOR (Fortran).
  Pour calculer C = A * B en ROW-MAJOR (C/C++), on utilise :
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                  &alpha, d_b, N, d_a, N, &beta, d_c, N)
  => calcule effectivement C^T = B^T * A^T, ce qui en row-major donne C = A*B
*/

void benchmark_cublas(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);

    vector<float> h_a(N*N), h_b(N*N), h_c(N*N);
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Création du handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, h2d_done, kernel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&kernel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_done, 0);

    float alpha = 1.0f, beta = 0.0f;

    // C = A * B  (en tenant compte du column-major de cuBLAS)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_b, N,   // B en col-major = B^T en row-major
                d_a, N,   // A en col-major = A^T en row-major
                &beta,
                d_c, N);

    cudaEventRecord(kernel_done, 0);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float h2d, kernel, d2h, total;
    cudaEventElapsedTime(&h2d,    start,       h2d_done);
    cudaEventElapsedTime(&kernel, h2d_done,    kernel_done);
    cudaEventElapsedTime(&d2h,    kernel_done, stop);
    cudaEventElapsedTime(&total,  start,       stop);

    float gflops = (2.0f * N/1000.0f * N/1000.0f * N/1000.0f) * 1000.0f / kernel;

    printf("\n=== cuBLAS (SGEMM) N=%d ===\n", N);
    printf("  Host->Device : %.3f ms\n", h2d);
    printf("  Kernel       : %.3f ms  (%.2f GFLOPS)\n", kernel, gflops);
    printf("  Device->Host : %.3f ms\n", d2h);
    printf("  TOTAL        : %.3f ms\n", total);

    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(h2d_done);
    cudaEventDestroy(kernel_done); cudaEventDestroy(stop);
}

// ============================================================
// PARTIE 2-2 : Tensor Cores (WMMA)
// Nécessite sm_70 ou supérieur (V100, T4, A100, RTX 2080+)
// Tile size : WMMA opère sur des tuiles 16x16
// ============================================================

// Taille des tuiles WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Taille d'un bloc (plusieurs tuiles WMMA par bloc)
#define BLOCK_SIZE 128  // threads par bloc (doit être multiple de 32)
#define WARP_SIZE  32

/*
  Principe :
  - Chaque WARP (32 threads) effectue une multiplication de tuile 16x16x16
    en une seule instruction WMMA.
  - Les matrices A et B sont en half precision (fp16) pour exploiter les Tensor Cores.
  - L'accumulation se fait en float32 pour la précision.
  
  Étapes :
  1. Charger des tuiles 16x16 de A et B en shared memory (fp16)
  2. Appeler wmma::load_matrix_sync
  3. Appeler wmma::mma_sync  (c'est l'instruction Tensor Core)
  4. wmma::store_matrix_sync vers C (fp32)
*/

__global__ void convertFp32ToFp16(half* out, const float* in, int n);

__global__ void matmulTensorCore(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  float*      __restrict__ C,
                                  int N) {
    // Chaque warp calcule une tuile WMMA_M x WMMA_N de C
    int warp_id   = (threadIdx.x + threadIdx.y * blockDim.x) / WARP_SIZE;
    int lane_id   = threadIdx.x % WARP_SIZE;

    // Nombre de warps par bloc (en x)
    int warps_per_block_x = blockDim.x / WARP_SIZE;

    // Coordonnée de la tuile dans la grille
    int tile_row = (blockIdx.y * (blockDim.y) + (warp_id / warps_per_block_x)) * WMMA_M;
    int tile_col = (blockIdx.x * warps_per_block_x + (warp_id % warps_per_block_x)) * WMMA_N;

    if (tile_row >= N || tile_col >= N) return;

    // Fragments WMMA
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              c_frag;

    // Initialiser accumulateur à zéro
    wmma::fill_fragment(c_frag, 0.0f);

    // Boucle sur les tuiles K
    for (int k = 0; k < N; k += WMMA_K) {
        // Charger les tuiles A et B
        if (tile_row < N && k + WMMA_K <= N) {
            wmma::load_matrix_sync(a_frag, A + tile_row * N + k, N);
        }
        if (tile_col < N && k + WMMA_K <= N) {
            wmma::load_matrix_sync(b_frag, B + k * N + tile_col, N);
        }
        // Multiplication-accumulation sur Tensor Core
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Stocker le résultat
    wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
}

void benchmark_tensorcore(int N) {
    size_t bytes_fp32 = (size_t)N * N * sizeof(float);
    size_t bytes_fp16 = (size_t)N * N * sizeof(half);

    // Initialisation en float sur CPU
    vector<float> h_a(N*N), h_b(N*N), h_c(N*N);
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

    // Convertir en half (fp16) sur GPU
    float *d_a_fp32, *d_b_fp32;
    half  *d_a_fp16, *d_b_fp16;
    float *d_c;

    cudaMalloc(&d_a_fp32, bytes_fp32);
    cudaMalloc(&d_b_fp32, bytes_fp32);
    cudaMalloc(&d_a_fp16, bytes_fp16);
    cudaMalloc(&d_b_fp16, bytes_fp16);
    cudaMalloc(&d_c,      bytes_fp32);

    // Kernel de conversion float -> half
    // On utilise cuBLAS pour simplifier la conversion
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, h2d_done, kernel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&kernel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Copier fp32 vers device
    cudaMemcpy(d_a_fp32, h_a.data(), bytes_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp32, h_b.data(), bytes_fp32, cudaMemcpyHostToDevice);

    // Conversion fp32 -> fp16 via cuBLAS
    // (utilisation du kernel inline pour la conversion)
    // Kernel de conversion simple
    auto convert = [&](float* src, half* dst, int n) {
        // Lancement d'un kernel lambda de conversion
        // cudaMemcpy ne peut pas faire la conversion, on utilise un kernel
    };

    // Kernel de conversion fp32 -> fp16
    // Lancé séparément (non compté dans le temps kernel Tensor Core)
    {
        int threads = 256;
        int blocks  = ((size_t)N * N + threads - 1) / threads;
        convertFp32ToFp16<<<blocks, threads>>>(d_a_fp16, d_a_fp32, N * N);
        convertFp32ToFp16<<<blocks, threads>>>(d_b_fp16, d_b_fp32, N * N);
    }

    cudaEventRecord(h2d_done, 0);

    // Configuration de la grille pour WMMA
    // Chaque bloc contient (BLOCK_SIZE) threads = (BLOCK_SIZE/WARP_SIZE) warps
    // Chaque warp calcule une tuile 16x16
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;   // ex: 128/32 = 4 warps
    int tiles_per_block = warps_per_block;
    dim3 block(WARP_SIZE, warps_per_block);
    dim3 grid((N + WMMA_N - 1) / WMMA_N / warps_per_block,
              (N + WMMA_M - 1) / WMMA_M);

    // NOTE : La conversion float->half est nécessaire avant ce kernel
    // Dans ce code, supposons que d_a_fp16 et d_b_fp16 sont déjà remplis
    matmulTensorCore<<<grid, block>>>(d_a_fp16, d_b_fp16, d_c, N);

    cudaEventRecord(kernel_done, 0);
    cudaMemcpy(h_c.data(), d_c, bytes_fp32, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float h2d, kernel, d2h, total;
    cudaEventElapsedTime(&h2d,    start,       h2d_done);
    cudaEventElapsedTime(&kernel, h2d_done,    kernel_done);
    cudaEventElapsedTime(&d2h,    kernel_done, stop);
    cudaEventElapsedTime(&total,  start,       stop);

    float gflops = (2.0f * N/1000.0f * N/1000.0f * N/1000.0f) * 1000.0f / kernel;

    printf("\n=== Tensor Cores (WMMA) N=%d ===\n", N);
    printf("  Host->Device : %.3f ms\n", h2d);
    printf("  Kernel       : %.3f ms  (%.2f GFLOPS)\n", kernel, gflops);
    printf("  Device->Host : %.3f ms\n", d2h);
    printf("  TOTAL        : %.3f ms\n", total);

    cublasDestroy(handle);
    cudaFree(d_a_fp32); cudaFree(d_b_fp32);
    cudaFree(d_a_fp16); cudaFree(d_b_fp16); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(h2d_done);
    cudaEventDestroy(kernel_done); cudaEventDestroy(stop);
}

// ============================================================
// Kernel de conversion float -> half (utilisé avant Tensor Core)
// ============================================================
__global__ void convertFp32ToFp16(half* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ============================================================
// MAIN
// ============================================================

int main() {
    int N = 8192;

    printf("============================================================\n");
    printf("  PARTIE 2 - cuBLAS & Tensor Cores  N=%d\n", N);
    printf("============================================================\n");

    benchmark_cublas(N);

    // Tensor Cores : nécessite sm_70+
    // Si votre GPU ne supporte pas WMMA, commentez la ligne suivante
    benchmark_tensorcore(N);
    printf("\n[INFO] Tensor Core benchmark : décommenter dans main() si GPU >= Volta\n");

    return 0;
}

/*
============================================================
  COMMENTAIRES PARTIE 2 (pour le rapport)
============================================================

--- 2-1 : cuBLAS ---
cuBLAS est la bibliothèque d'algèbre linéaire optimisée par NVIDIA.
Elle utilise :
  - Des noyaux assembleur hautement optimisés
  - Le tiling automatique (shared memory + registres)
  - L'accès mémoire coalesced
  - Les instructions vectorielles SIMD
  - Sur GPU récent : utilise automatiquement les Tensor Cores

Performance attendue vs naïve :
  - cuBLAS est ~10x à 50x plus rapide que MatmulXrow
  - Pour N=8192 : cuBLAS peut atteindre >10 TFLOPS sur A100

--- 2-2 : Tensor Cores ---
Principe :
  - Introduits avec l'architecture Volta (2017)
  - Ce sont des unités matérielles dédiées dans chaque SM
  - Effectuent D = A*B + C sur des tuiles 16x16 en UNE seule instruction
  - En FP16 (half precision) pour A et B, FP32 pour C et D
  - Un Tensor Core effectue 64 opérations FMA en 1 cycle d'horloge
    vs. 1 FMA/cycle pour une unité CUDA FP32 normale

Bénéfices :
  - Débit 8x supérieur aux CUDA cores FP32 pour la matmul
  - Exemple : A100 = 312 TFLOPS (Tensor FP16) vs 19.5 TFLOPS (FP32)
  - Crucial pour le Deep Learning (training & inference)

Implémentation via WMMA API :
  - wmma::fragment : représente une tuile en registres distribués sur le warp
  - wmma::load_matrix_sync : charge depuis global/shared memory
  - wmma::mma_sync : exécute l'instruction Tensor Core
  - wmma::store_matrix_sync : écrit le résultat

Comparaison attendue (N=8192) :
  Kernel naïf   :  ~50  GFLOPS
  MatmulYrow    :  ~100 GFLOPS  (meilleur accès mémoire)
  cuBLAS        :  ~5000 GFLOPS
  Tensor Cores  :  ~10000 GFLOPS (FP16)
============================================================
*/
