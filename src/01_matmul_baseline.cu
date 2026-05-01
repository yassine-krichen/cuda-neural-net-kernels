// ============================================================
// LAB 2 - PARTIE 1 : Multiplication de Matrices (Float & Int)
// MatmulXrow et MatmulYrow pour N=8192
// ============================================================

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// ============================================================
// KERNELS FLOAT
// ============================================================

__global__ void matrixMulXrow_float(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = tmp;
    }
}

__global__ void matrixMulYrow_float(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = tmp;
    }
}

// ============================================================
// KERNELS INT (entier)
// ============================================================

__global__ void matrixMulXrow_int(const int* a, const int* b, int* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int tmp = 0;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = tmp;
    }
}

__global__ void matrixMulYrow_int(const int* a, const int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int tmp = 0;
        for (int k = 0; k < N; k++) {
            tmp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = tmp;
    }
}

// ============================================================
// Fonction utilitaire : mesure du temps
// ============================================================

struct TimingResult {
    float host2dev;
    float kernel;
    float dev2host;
    float total;
};

// ============================================================
// Benchmark FLOAT
// ============================================================

TimingResult benchmark_float(const char* name, bool useXrow, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);

    vector<float> h_a(N * N), h_b(N * N), h_c(N * N);
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaEvent_t start, h2d_done, kernel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&kernel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_done, 0);

    int THREADS = 32;
    int BLOCKS  = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    if (useXrow)
        matrixMulXrow_float<<<blocks, threads>>>(d_a, d_b, d_c, N);
    else
        matrixMulYrow_float<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaEventRecord(kernel_done, 0);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    TimingResult t;
    cudaEventElapsedTime(&t.host2dev,  start,       h2d_done);
    cudaEventElapsedTime(&t.kernel,    h2d_done,    kernel_done);
    cudaEventElapsedTime(&t.dev2host,  kernel_done, stop);
    cudaEventElapsedTime(&t.total,     start,       stop);

    float gflops = (2.0f * N / 1000.0f * N / 1000.0f * N / 1000.0f) * 1000.0f / t.kernel;

    printf("\n=== [FLOAT] Kernel: %s ===\n", name);
    printf("  Host->Device : %.3f ms\n", t.host2dev);
    printf("  Kernel       : %.3f ms  (%.2f GFLOPS)\n", t.kernel, gflops);
    printf("  Device->Host : %.3f ms\n", t.dev2host);
    printf("  TOTAL        : %.3f ms\n", t.total);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(h2d_done);
    cudaEventDestroy(kernel_done); cudaEventDestroy(stop);
    return t;
}

// ============================================================
// Benchmark INT
// ============================================================

TimingResult benchmark_int(const char* name, bool useXrow, int N) {
    size_t bytes = (size_t)N * N * sizeof(int);

    vector<int> h_a(N * N), h_b(N * N), h_c(N * N);
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaEvent_t start, h2d_done, kernel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&kernel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_done, 0);

    int THREADS = 32;
    int BLOCKS  = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    if (useXrow)
        matrixMulXrow_int<<<blocks, threads>>>(d_a, d_b, d_c, N);
    else
        matrixMulYrow_int<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaEventRecord(kernel_done, 0);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    TimingResult t;
    cudaEventElapsedTime(&t.host2dev,  start,       h2d_done);
    cudaEventElapsedTime(&t.kernel,    h2d_done,    kernel_done);
    cudaEventElapsedTime(&t.dev2host,  kernel_done, stop);
    cudaEventElapsedTime(&t.total,     start,       stop);

    float gflops = (2.0f * N / 1000.0f * N / 1000.0f * N / 1000.0f) * 1000.0f / t.kernel;

    printf("\n=== [INT] Kernel: %s ===\n", name);
    printf("  Host->Device : %.3f ms\n", t.host2dev);
    printf("  Kernel       : %.3f ms  (%.2f GFLOPS)\n", t.kernel, gflops);
    printf("  Device->Host : %.3f ms\n", t.dev2host);
    printf("  TOTAL        : %.3f ms\n", t.total);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(h2d_done);
    cudaEventDestroy(kernel_done); cudaEventDestroy(stop);
    return t;
}

// ============================================================
// MAIN
// ============================================================

int main() {
    int N = 8192;

    printf("============================================================\n");
    printf("  PARTIE 1 - Multiplication de Matrices  N=%d\n", N);
    printf("============================================================\n");

    // ---- FLOAT ----
    TimingResult xrow_f = benchmark_float("MatmulXrow", true,  N);
    TimingResult yrow_f = benchmark_float("MatmulYrow", false, N);

    // ---- INT ----
    TimingResult xrow_i = benchmark_int("MatmulXrow", true,  N);
    TimingResult yrow_i = benchmark_int("MatmulYrow", false, N);

    // ---- Tableau récapitulatif ----
    printf("\n============================================================\n");
    printf("  TABLEAU RECAPITULATIF\n");
    printf("  %-16s  %12s  %12s\n", "Kernel", "FLOAT(ms)", "INT(ms)");
    printf("  %-16s  %12.3f  %12.3f\n", "MatmulXrow Total", xrow_f.total, xrow_i.total);
    printf("  %-16s  %12.3f  %12.3f\n", "MatmulYrow Total", yrow_f.total, yrow_i.total);
    printf("============================================================\n");

    return 0;
}

/*
============================================================
  COMMENTAIRES / ANALYSE (à inclure dans le rapport)
============================================================

1. MatmulXrow vs MatmulYrow :
   - Dans MatmulXrow : row = blockIdx.x*blockDim.x + threadIdx.x
     => Les threads d'un même warp ont des 'row' consécutifs mais la même 'col'
     => Accès à b[k*N+col] : MEME colonne pour tous les threads => NON coalesced
   - Dans MatmulYrow : col = blockIdx.x*blockDim.x + threadIdx.x
     => Les threads d'un même warp ont des 'col' consécutives
     => Accès à b[k*N+col] : colonnes consécutives => COALESCED => plus rapide

2. Float vs Int :
   - Sur GPU moderne (Pascal, Turing, Ampere) : les unités FP32 et INT32 sont
     séparées. Les deux types bénéficient de débit similaire (1 op/cycle).
   - Sur architectures plus anciennes (Kepler, Maxwell) : INT32 peut être
     légèrement plus lent car partagé avec FP32.
   - La différence principale : pas de FMA (Fused Multiply-Add) pour INT32
     sur la plupart des architectures => légère pénalité possible.
   - Les temps de transfert mémoire sont identiques (sizeof(int)==sizeof(float)==4).
============================================================
*/
