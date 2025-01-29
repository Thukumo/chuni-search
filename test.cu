#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;
#define threads_per_block 32
#define typeof_memo int

__global__ void calc_score(int jc, int j, int max_notes, typeof_memo *memo, typeof_memo *points) {
    int a = blockIdx.y, m = blockIdx.z*threads_per_block + threadIdx.x; //mだけグリッドzとブロック内で分けてる
    int idx = blockIdx.y * gridDim.z * threads_per_block + m; // 修正: blockDim.z -> gridDim.z
    if (max_notes < jc + j + a + m || !(jc + j + a + m))
    {
        points[idx] = -1;
        return;
    }
    //points[idx] = memo[(size_t)((jc*1.01f + j + a * 0.5f) * 1000000 / (jc + j + a + m))] + memo[jc] + memo[j] + memo[a] + memo[m];
    //points[idx] =memo[(size_t)((jc*1.01f + j + a * 0.5f) * 1000000 / (jc + j + a + m))];
    points[idx] = ((jc*1.01f + j + a * 0.5f) * 1000000 / (jc + j + a + m));
    return;
}

int main() //これは今後のテスト用に残しておく
{
    int test = 100;
    dim3 grid(1, test+1, (test+1)/threads_per_block + !!(test%threads_per_block));
    dim3 block(threads_per_block);
    typeof_memo *host_memo = new typeof_memo[1010000 + 1], *memo, *points;
    cudaMallocManaged(&points, sizeof(typeof_memo) *grid.x * grid.y * grid.z * threads_per_block);
    #pragma omp parallel for
    for (int i = 0; i <= 1010000; i++) {
        auto s_i = to_string(i);
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') host_memo[i]++;
    }
    cudaMalloc(&memo, sizeof(typeof_memo) * (1010000 + 1));
    cudaMemcpy(memo, host_memo, sizeof(typeof_memo) * (1010000 + 1), cudaMemcpyHostToDevice);
    delete[] host_memo;
    calc_score << <grid, block >> > (3898 - 56 - 18, 56, 40000, memo, points);
    cudaDeviceSynchronize();
    for (int a = 0; a < grid.y; a++)
    {
        for (int m = 0; m < grid.z*threads_per_block; m++)
        {
            int idx = a * grid.z * threads_per_block + m;
            if (points[idx] >= 0) {
                if(1007500 <= points[idx]) cout << "(" << a << ", " << m << "): " << points[idx] << endl;
            }
        }
    }
}
