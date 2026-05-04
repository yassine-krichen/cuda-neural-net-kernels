<div align="center">

# 🚀 Codesign Lab 2: CUDA Programming
### High-Performance GPU Kernels

[![NVIDIA GPU](https://img.shields.io/badge/GPU-RTX%203050-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-MSVC%20%2F%20NVCC-00599C?style=for-the-badge&logo=c%2B%2B)](https://isocpp.org/)

</div>

This repository contains the results and implementation of the **CUDA Programming Lab**, focusing on GPU-accelerated computing for General Purpose Applications (GPGPU), spanning baseline memory coalescence, cuBLAS integration, Tensor Core operations, and custom Neural Network Sum Reductions.

---

## 📂 Project Structure

```text
├── src/                        # Kernel implementations
│   ├── 01_matmul_baseline.cu     # Naive row/col Matrix Multiplication (Float/Int)
│   ├── 02_matmul_advanced.cu     # cuBLAS & Tensor Core Implementations
│   ├── 03_reduction_naive.cu     # Naive Sum Reduction for Neural Networks
│   └── 04_reduction_optimized.cu # Optimized Sum Reductions (Warp Shuffle, Unrolling)
├── scripts/                    # Build and execution scripts
│   ├── build_and_run_all.bat     # Compile and execute the full pipeline
│   ├── run_part2.bat             # Fallback/specific configuration scripts
│   └── run_part2_fixed.bat
├── third_party/                # Third-party libraries and wheels (not source)
│   └── pypdf/                   # bundled pypdf wheel/extracted package
├── results/                    # Benchmark outputs and raw results
└── docs/                       # Problem statements, summaries, and reports
```

---

## ⚙️ How to Build and Run

To ensure maximum compatibility, this project relies on **NVCC** (The NVIDIA CUDA Compiler) and the standard MSVC compiler on Windows. 

### Prerequisites
1. **NVIDIA GPU** with Compute Capability `7.0` or higher (Tensor Core support requires `7.0+`, Ampere `8.6` recommended).
2. **CUDA Toolkit** installed (Code was verified on CUDA `13.2`).
3. **Microsoft Visual Studio** (C++ Build Tools) installed and mapped to the system `%PATH%`.

### Compilation & Execution
Open a **Developer Command Prompt for VS** (or any terminal where `nvcc` is in the PATH), and execute the global build script. This will compile the kernels and run the default benchmarks.

```bat
cd path\to\repo
.\scripts\build_and_run_all.bat
```

You can run individual scripts from `scripts\` as needed:

```bat
.\scripts\build_and_run_all.bat    # build and run the full pipeline
.\scripts\run_part2.bat           # run Part 2 (cuBLAS / TensorCore tests)
.\scripts\run_part2_fixed.bat     # alternate configuration for Part 2
```

Alternatively compile a single source file manually (example linking cuBLAS):
```bat
nvcc src\04_reduction_optimized.cu -o bin_opt -lcublas
.\bin_opt.exe
```

Note: third-party python libs are placed under `third_party/` (e.g. `third_party/pypdf`). These are not required to compile the CUDA code but are included for reference or local utilities.

---

## 📊 Benchmark Showcase

**Hardware Profile:**
- **GPU:** NVIDIA GeForce RTX 3050 (Ampere Architecture)
- **Problem Size $N$:** `8192 x 8192` (256 MB per matrix)

### 1. Matrix Multiplication: Memory Coalescence
Naive baseline vs Column-major memory access:
- **MatmulXrow** (Non-coalesced): 18,583.3 ms (59.17 GFLOPS)
- **MatmulYrow** (Coalesced): 2,537.4 ms (433.32 GFLOPS) -> **~7.3x Speedup**
> *Note: Float32 and Int32 performed identically on Ampere due to concurrent separated datapaths and shared memory-bound constraints.*

### 2. High-Performance Math Libraries & Tensor Cores
- **cuBLAS (SGEMM):** 321.7 ms (3,417.53 GFLOPS) **(57.7x vs Naive)**
- **Tensor Cores (WMMA):** 77.0 ms (14,268.05 GFLOPS) **(241.1x vs Naive)**

### 3. Neural Network Sum Reduction
Optimizing an 8192x8192 Sigmoid Hidden Layer. Performance scales based on warp occupancy and register bandwidth:
| Technique | Avg Time (ms) | Speedup |
| :--- | :---: | :---: |
| Naive Reduction | 15.535 | 1.00x |
| Interleaved Addressing | 2.281 | 6.81x |
| Loop Unrolling | 1.966 | 7.90x |
| **Warp Shuffle** | **1.735** | **8.95x** |

## 🚀 Quick Start
Run the test suite using the provided bat script:
```powershell
.\scripts\01_build_and_run_all.bat
```
*(Ensure MSVC and NVCC are configured in your `%PATH%`)*
