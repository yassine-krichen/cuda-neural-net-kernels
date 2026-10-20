# cuda-neural-net-kernels

*CUDA kernels for matrix multiplication and neural-network sum reduction, benchmarked on an RTX 3050.*

![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)
![GPU](https://img.shields.io/badge/GPU-RTX%203050-76B900)
![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-lightgrey)

Lab 2 of the CODESIGN course (GL3, INSAT). The goal is to see how much of a GPU's throughput comes from how memory is accessed and which hardware path is used, so each kernel is written in a naive form first and then improved one technique at a time. Every variant is timed on the same problem size, which makes the effect of each technique visible.

## What is in it

| Part | Kernels | Technique |
| :--- | :--- | :--- |
| 1. Matrix multiplication | `matrixMulXrow`, `matrixMulYrow` (float and int) | Memory coalescing: the same arithmetic with two thread-to-element mappings |
| 2. Libraries and hardware | `cublasSgemm`, `matmulTensorCore` | cuBLAS SGEMM, and Tensor Cores through the WMMA API (16x16 tiles, FP16 inputs, FP32 accumulation) |
| 3. Sum reduction | naive, interleaved, unrolled, warp shuffle | Reduction for a sigmoid hidden layer (8192 neurons x 8192 inputs): removing warp divergence, unrolling the last warp, and `__shfl_down_sync` |

## Layout

```text
src/
  01_matmul_baseline.cu       coalesced vs non-coalesced matmul, float and int
  02_matmul_advanced.cu       cuBLAS SGEMM and WMMA Tensor Core matmul
  03_reduction_naive.cu       naive sum reduction for the hidden layer
  04_reduction_optimized.cu   interleaved, unrolled and warp-shuffle reductions
  99_legacy_kernel_original.cu  first version of the matmul kernels, kept for reference
scripts/                      Windows .bat files that build and run each part
results/                      raw benchmark output and a formatted summary
lab2_report.md                the lab report
docs/                         the assignment and the CUDA introduction guide
```

## How it works

- **Coalescing.** In `matrixMulXrow` neighbouring threads of a warp read addresses far apart in memory. `matrixMulYrow` maps threads so a warp reads consecutive addresses, which lets the hardware serve a warp with few memory transactions.
- **cuBLAS.** `cublasSgemm` is called on the same matrices. cuBLAS is column-major, so the call swaps the operands to get a row-major product.
- **Tensor Cores.** Inputs are converted from FP32 to FP16 on the device (`convertFp32ToFp16`), then each warp computes 16x16 tiles with `nvcuda::wmma`, accumulating in FP32.
- **Reductions.** The naive kernel halves the active threads with a strided pattern that leaves warps divergent. The interleaved version keeps active threads contiguous. The unrolled version removes `__syncthreads()` from the last 32 steps. The warp-shuffle version finishes inside a warp through registers with `__shfl_down_sync`, without shared memory. The benchmark checks each result against the naive output and prints `[VALID]`.

## Build and run

Requirements: an NVIDIA GPU with compute capability 7.0 or higher (Tensor Cores), the CUDA Toolkit (developed on 13.2) and the MSVC build tools on Windows.

```bat
:: from a Developer Command Prompt for VS
scripts\01_build_and_run_all.bat
```

The scripts contain absolute paths to the Visual Studio and CUDA install directories used on the author's machine; edit them first. A single file can also be built directly:

```bat
nvcc src\02_matmul_advanced.cu -o part2 -lcublas
part2.exe
```

## Measured results

RTX 3050 (Ampere), CUDA 13.2, MSVC, N = 8192 (256 MB per matrix). Raw output is in `results/raw_benchmarks.txt`, the formatted version in `results/showcase_results.md`. Each figure is a single run for matmul, and the mean of 10 iterations for the reductions.

**Matrix multiplication**

| Kernel | Time (ms) | GFLOPS |
| :--- | ---: | ---: |
| Non-coalesced, float | 18,583.3 | 59.17 |
| Coalesced, float | 2,537.4 | 433.32 |
| Non-coalesced, int | 18,477.3 | 59.51 |
| Coalesced, int | 2,357.6 | 466.37 |
| cuBLAS SGEMM | 321.7 | 3,417.53 |
| Tensor Cores (WMMA, FP16 in / FP32 acc) | 77.0 | 14,268.05 |

Coalescing alone gives about 7.3x. The 57.7x (cuBLAS) and 241.1x (Tensor Cores) figures in the report are relative to the non-coalesced kernel, so they overstate the gain over a reasonable hand-written baseline. The Tensor Core path uses FP16 inputs and its error against FP32 was not measured.

**Sum reduction (8192 x 8192, sigmoid)**

| Variant | Mean time (ms) | Speedup |
| :--- | ---: | ---: |
| Naive | 15.535 | 1.00x |
| Interleaved addressing | 2.281 | 6.81x |
| Loop unrolling | 1.966 | 7.90x |
| Warp shuffle | 1.735 | 8.95x |

## Limitations

No CPU baseline, no run-to-run variance, no profiler (Nsight) data, and no shared-memory tiled matmul to compare against cuBLAS. These are the obvious next steps.

## License

**Source-available, not open source.** Published under the
[PolyForm Strict License 1.0.0](LICENSE) so you can read the code, run it, and learn
from it for personal, research, or evaluation use.

You may **not** use it commercially, redistribute it, rebrand it, or publish modified
versions without written permission. Copyright (c) 2026 Yassine Krichen. All rights
not granted by the license are reserved.

For commercial licensing or collaboration: krichenyassine22@gmail.com
