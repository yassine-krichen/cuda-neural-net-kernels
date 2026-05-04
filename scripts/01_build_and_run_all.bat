@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"

echo --- PART 1 ---
%NVCC% src\01_matmul_baseline.cu -o part1 -O3
part1.exe

echo --- PART 2 ---
%NVCC% src\02_matmul_advanced.cu -o part2 -lcublas -arch=compute_86 -code=sm_86 -O3
part2.exe

echo --- PART 3-1 ---
%NVCC% src\03_reduction_naive.cu -o part3_1 -O3
part3_1.exe

echo --- PART 3-2 ---
%NVCC% src\04_reduction_optimized.cu -o part3_2 -O3
part3_2.exe
