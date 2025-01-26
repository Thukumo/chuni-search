@nvcc -O3 4.cu -o 4.exe -Xcompiler "/O2 /GA /openmp /favor:INTEL64 /Qvec-report:1 /Qpar /Qpar-report:1"
