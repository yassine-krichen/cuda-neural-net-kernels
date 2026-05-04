# 🚀 CUDA Lab 2: Sum Reduction & Matrix Multiplication 
**Performance Benchmark Showcase**

**Hardware Profile:**
- **GPU:** NVIDIA GeForce RTX 3050 (Ampere Architecture)
- **CUDA Version:** 13.2
- **Compiler:** MSVC (Visual Studio 2026) & NVCC
- **Matrix Size ($N$):** 8192 x 8192 (256 MB per matrix)

---

## 📊 Part 1: Naive Matrix Multiplication (Float vs. Int)

In this part, we compare the naive row-by-row (`MatmulXrow`) and column-by-column (`MatmulYrow`) approaches for both `Float` and `Integer` matrices.

### Results
| Kernel Version | Data Type | Kernel Time (ms) | Performance (GFLOPS) | Memory Access |
| :--- | :---: | :---: | :---: | :--- |
| **MatmulXrow** | `FLOAT` | 18,583.3 | 59.17 | Non-coalesced |
| **MatmulYrow** | `FLOAT` | 2,537.4 | 433.32 | **Coalesced** |
| **MatmulXrow** | `INT` | 18,477.3 | 59.51 | Non-coalesced |
| **MatmulYrow** | `INT` | 2,357.6 | 466.37 | **Coalesced** |

> [!TIP]
> **Key Takeaways:**
> 1. **Memory Coalescence is King:** `MatmulYrow` is roughly **7.3x faster** than `MatmulXrow`. By rearranging the loops so that threads in a warp access consecutive memory addresses, we achieve *Memory Coalescence*, maximizing memory bandwidth.
> 2. **Float vs Int:** On the modern Ampere architecture of the RTX 3050, `Float` and `Int` operations perform almost identically because the FP32 and INT32 units are separated and can execute concurrently with similar throughput.

---

## 🏎️ Part 2: High-Performance Libraries & Tensor Cores

Here we abandon the naive approach and use professional tools: NVIDIA's highly-optimized **cuBLAS** library, and the **Tensor Cores** (via WMMA) which are dedicated hardware accelerators.

### Results
| Implementation | Kernel Time (ms) | Performance (GFLOPS) | Speedup vs Naive Xrow |
| :--- | :---: | :---: | :---: |
| **MatmulXrow** (Baseline) | 18,583.3 | 59.17 | 1.0x |
| **cuBLAS (SGEMM)** | 321.7 | 3,417.53 | **57.7x** |
| **Tensor Cores (WMMA)** | **77.0** | **14,268.05** | **241.1x** |

> [!IMPORTANT]
> **Why Tensor Cores are revolutionary:**
> cuBLAS uses deep assembly optimizations to run 57x faster than our naive code. However, **Tensor Cores** crush even cuBLAS by operating on 16x16 matrix blocks (`fp16` inputs, `fp32` accumulation) in a *single clock cycle*. This yields a staggering **14.2 TFLOPS**, making it ~241x faster than the baseline!

---

## 🧠 Part 3: Neural Network Sum Reduction

For the neural network hidden layer, we must sum 8,192 elements per neuron. We started with a Naive reduction and applied three distinct optimization strategies. Below is the final *verified* output from the benchmark suite:

```text
============================================================
  PARTIE 3-2 : Optimisations Sum Reduction
  Input size  : 8192
  Hidden size : 8192
  Block size  : 1024
============================================================

--- Temps moyen sur 10 iterations ---

  Naive (reference)                   : 15.535 ms [VALID]
  Amelioration 1 : Entrelacee         : 2.281 ms [VALID]
  Amelioration 2 : Warp Shuffle       : 1.735 ms [VALID]
  Amelioration 3 : Unrolling          : 1.966 ms [VALID]

============================================================
  TABLEAU RECAPITULATIF DES PERFORMANCES
  Kernel                               Temps(ms)   Speedup
  Naive (reference)                      15.535      1.00x
  Amelioration 1 : Entrelacee             2.281      6.81x
  Amelioration 2 : Warp Shuffle           1.735      8.95x
  Amelioration 3 : Unrolling              1.966      7.90x
============================================================
```

*(Note: Times are averaged over 10 iterations for precision)*

### Results
| Optimization Level | Technique Used | Avg Time (ms) | Speedup vs Naive |
| :--- | :--- | :---: | :---: |
| **Baseline** | Naive Reduction | 15.535 | 1.00x |
| **Optimization 1** | Interleaved Addressing | 2.281 | 6.81x |
| **Optimization 3** | Loop Unrolling | 1.966 | 7.90x |
| **Optimization 2** | **Warp Shuffle** (`__shfl_down_sync`) | **1.735** | **8.95x** |

> [!TIP]
> **Optimization Breakdown:**
> - **Naive:** Slowest due to *warp divergence* (half the threads in a warp are idle).
> - **Interleaved:** Fixes warp divergence by keeping active threads contiguous. (+36% speedup)
> - **Unrolling:** Removes costly `__syncthreads()` barriers during the final 32-thread reduction steps. (+61% speedup)
> - **Warp Shuffle:** The ultimate optimization. It bypasses shared memory completely, allowing threads to share data directly via registers in a single clock cycle. (+87% speedup)
