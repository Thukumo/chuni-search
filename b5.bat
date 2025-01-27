@nvcc -O3 5.cu -o 5.exe -Xcompiler "/O2 /GA /openmp /favor:INTEL64 /Qvec-report:1 /Qpar /Qpar-report:1"
