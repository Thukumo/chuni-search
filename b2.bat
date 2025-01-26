@nvcc -O3 2.cu -o 2.exe -Xcompiler "/O2 /GA /openmp /favor:INTEL64 /Qvec-report:1 /Qpar /Qpar-report:1"
