# CUDA Programming Lab 2 - Full Report
**Date:** May 10, 2026
**Course:** CODESIGN – GL3

---

## Part 1: Matmul Baseline (Float vs Int)

### 1-2. Kernel Time for Float and Integer Arrays
| Kernel | Runtime (FLOAT) | Runtime (INT) |
| :--- | :--- | :--- |
| **MatmulXrow** | 18,583.3 ms | 18,477.3 ms |
| **MatmulYrow** | 2,537.4 ms | 2,357.6 ms |

### Questions and Interpretation
**Q: Based on your NVIDIA device characteristics, do you think that the implementation of the same kernel with integer Matrices will give the same results? Explain.**
_Answer:_ Yes. Modern NVIDIA streaming multiprocessors (like the Ampere architecture in the RTX 3050) feature separated pipelines for `FP32` and `INT32` operations. Both units are capable of completing instructions symmetrically in most basic ALU computations. Therefore, math throughput limits are practically identical.

**Q: Compare the performances of the Float and Integer implementations and interpret.**
_Answer:_ The performance times are identical. This verifies our hypothesis that the `INT` performance will match `FLOAT`. Moreover, naïve matrix multiplication is highly **memory-bound**. The ~7.3x speedup moving from `MatmulXrow` to `MatmulYrow` stems from **Memory Coalescence**. In `MatmulYrow`, contiguous threads (within a warp) access contiguous memory addresses in the global memory, optimizing L1/L2 cache lines bounds and vastly improving global memory bandwidth compared to column-jumping in `MatmulXrow`.

---

## Part 2: High Performance Libraries (cuBLAS & TensorCores)

### 1. cuBLAS Implementation
NVIDIA's `cuBLAS` SGEMM handles matrix multiplication using highly optimized assembly code out of the box. 
**Performance:** The cuBLAS routine executed in **321.7 ms** (3,417 GFLOPS), yielding a speedup of roughly **57x** compared to our naive CPU-like baseline `MatmulXrow`.

### 2. Tensor Cores Operations
**Principle and Benefits:** Tensor Cores are fixed-function math units inside the Streaming Multiprocessor dedicated exclusively to Deep Learning matrix accumulations. Instead of standard scalar Fused-Multiply-Add (FMA) instructions, Tensor Cores execute WMMA (Warp-Level Matrix Multiply and Accumulate) working on 4x4 or 16x16 sub-blocks within a *single clock cycle*. This drastically cuts down instruction overhead, register usage, and cycles per pipeline.
**Performance:** Using Tensor Cores via hardware intrinsics lowered the execution time to an unbelievable **77.0 ms**. This translates to over **14.2 TFLOPS**, running about **4x faster than cuBLAS** and **241x faster than MatmulXrow**.

---

## Part 3: Sum Reduction inside Neural Networks

### Part 3-1: Naive Implementation
**3-1-A: Why is this implementation inefficient?**
The initial iteration is inefficient because of **Warp Divergence**. At each step of the reduction tree loop `tid % (2 * stride) == 0`, half of the active threads inside a warp branch off and remain idle. Because warps execute in lock-step mode, the SM wastes execution slots on inactive threads.

**3-1-B: Architecture Details**
- **Activation Function:** Following the reduction of a single neuron's weighted inputs, thread 0 of each block calculates the Sigmoid:
  `h[neuron] = 1.0f / (1.0f + expf(-sharedData[0]));`
- **Block Size Value:** We used a block size of **1024** (the CUDA ceiling per block). Since each neuron has 8,192 inputs, an active stride of 8 elements is looped per thread prior to entering the shared memory phase.
- **Kernel Invocation:** 
  `sumReductionNaive<<<8192, 1024, sharedMemSize>>>(d_xW, d_h, 8192);` (8,192 blocks targeting our 8,192 hidden layer neurons).

### Part 3-2: Optimizations

1. **Improvement 1: Interleaved Addressing**
   *Technique:* We adjusted the stride mechanism so the active threads remain contiguous (`sharedData[tid] += sharedData[tid + stride]`).
   *Why it improves performance:* It removes warp divergence in the early loops. The processor can pack all active threads tightly into the first warps instead of spreading out active modulo threads across every warp.
   *Speedup:* Increased speed by **6.81x**.

2. **Improvement 2: Loop Unrolling last warps**
   *Technique:* When the reduction targets the final 32 elements inside shared memory, synchronization arrays (`__syncthreads()`) become useless since all elements map to a single executing warp. We unroll the loop to avoid barrier overheads.
   *Why it improves performance:* Stops synchronization bottlenecks.
   *Speedup:* Increased speed by **7.90x** (over naive).

3. **Improvement 3: Warp Shuffle Instructions (`__shfl_down_sync`)**
   *Technique:* Bypass shared memory outright in favor of peer-to-thread register shifts.
   *Why it improves performance:* Writing to and executing blocks from shared memory is fast, but Registers are the absolute fastest tier of the hierarchy. Shuffling data directly intra-warp using `__shfl_down_sync` mitigates memory bounds entirely during the final phases of reduction.
   *Speedup:* Produced the quickest reduction time at **1.735 ms** (**8.95x vs Naive**).