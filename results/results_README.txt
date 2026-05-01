============================================================
  LAB 2 CUDA - GUIDE DE COMPILATION ET D'UTILISATION
============================================================

FICHIERS FOURNIS :
  part1_matmul.cu          -> Partie 1 : MatmulXrow & MatmulYrow (float + int)
  part2_cublas_tensor.cu   -> Partie 2 : cuBLAS & Tensor Cores
  part3_1_naive.cu         -> Partie 3-1 : Sum Reduction naïve + NN
  part3_2_optimized.cu     -> Partie 3-2 : Sum Reduction optimisée

============================================================
  COMPILATION
============================================================

--- Partie 1 ---
nvcc part1_matmul.cu -o part1 -O3
./part1

--- Partie 2 (cuBLAS) ---
nvcc part2_cublas_tensor.cu -o part2 -lcublas -O3
./part2

--- Partie 2 (Tensor Cores, GPU Volta/Turing/Ampere requis) ---
nvcc part2_cublas_tensor.cu -o part2 -lcublas -arch=sm_70 -O3
# Pour RTX 2080/3080 (Turing/Ampere) :
nvcc part2_cublas_tensor.cu -o part2 -lcublas -arch=sm_75 -O3

--- Partie 3-1 ---
nvcc part3_1_naive.cu -o part3_1 -O3
./part3_1

--- Partie 3-2 ---
nvcc part3_2_optimized.cu -o part3_2 -O3
./part3_2

============================================================
  ARCHITECTURE GPU MINIMALE
============================================================

Partie 1 & 3 : tout GPU CUDA (compute capability >= 3.0)
Partie 2 cuBLAS : tout GPU CUDA
Partie 2 Tensor Core : GPU Volta ou supérieur (V100, T4, RTX 2080+, A100)
  - sm_70 : Volta  (V100)
  - sm_75 : Turing (RTX 2080, T4)
  - sm_80 : Ampere (A100, RTX 3090)

Pour vérifier votre GPU :
  nvidia-smi
  deviceQuery (depuis les CUDA samples)

============================================================
  RÉSULTATS ATTENDUS (ordre de grandeur)
============================================================

PARTIE 1 (N=8192) :
  MatmulXrow Float : ~2000-5000 ms kernel
  MatmulYrow Float : ~1500-4000 ms kernel  (meilleur accès mémoire)
  MatmulXrow Int   : ~similaire float
  MatmulYrow Int   : ~similaire float

PARTIE 2 (N=8192) :
  cuBLAS       : ~10-50 ms  (10x-100x plus rapide que naïf)
  Tensor Cores : ~5-20 ms   (plus rapide que cuBLAS car FP16)

PARTIE 3 :
  Naïf           : ~50-200 ms
  Entrelacée     : ~25-100 ms  (2x speedup)
  Warp Shuffle   : ~15-60 ms   (3-4x speedup)
  Unrolling      : ~20-80 ms   (2.5x speedup)

============================================================
  NOTES IMPORTANTES
============================================================

1. Pour N=8192 en Partie 1, le verify_result() est DÉSACTIVÉ en mode INT
   car les overflows entiers rendent la vérification impossible.

2. Le Tensor Core kernel requiert que N soit multiple de 16.
   N=8192 = 512 * 16 => OK.

3. Les temps de transfert mémoire (H2D, D2H) peuvent dominer pour les
   petites matrices. Pour N=8192 :
   - Taille xW (Part 3) = 8192*8192*4 = 256 Mo
   - Transfert à ~10 GB/s PCIe 3.0 x16 => ~25 ms

4. Pour reproduire les résultats, fixer le seed aléatoire :
   srand(42);  // Dans le main()
============================================================
