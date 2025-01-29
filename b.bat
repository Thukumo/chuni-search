@nvcc -O3 %1.cu -o %1.exe -Xcompiler "/O2 /GA /openmp /favor:INTEL64 /Qvec-report:1 /Qpar /Qpar-report:1"
