@nvcc -O3 kernel.cu -o kernel.exe -Xcompiler "/O2 /GA /openmp /favor:INTEL64 /Qvec-report:1 /Qpar /Qpar-report:1"
