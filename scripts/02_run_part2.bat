@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin

echo --- PART 2 ---
%NVCC% part2_cublas_tensor.cu -o part2 -lcublas -arch=sm_86 -O3
part2.exe
