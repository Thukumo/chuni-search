#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;

#define typeof_memo int8_t
//ここでGPUメモリ(ホストメモリも)の使用量を調整
#define memory_usage_limit 1024 * 4 //MB
#define max_notes 4444

#define cudaDo_Check(err)\
{\
    cudaError_t tmp = err;\
    if (tmp)\
    {\
        cerr <<"!!!ERROR!!! " << endl;\
        cerr << "cudaError_t: " << tmp << " " << cudaGetErrorString(tmp) << " At " << __FILE__ << ":" << __LINE__ << endl;\
        cudaDeviceReset();\
        return -1;\
    }\
}
#define kernelCheck() cudaDo_Check(cudaGetLastError())

#define mul_elem(bl) (bl.x * bl.y * bl.z)

#define get_info()\
{\
    cudaDeviceProp prop;\
    cudaGetDeviceProperties(&prop, 0);\
    cout << "Device name: " << prop.name << endl;\
    cout << "Memory: " << prop.totalGlobalMem / 1024 / 1024 << "MB" << endl;\
    cout << "Threads per block: " << prop.maxThreadsPerBlock << endl;\
    cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << "KB" << endl;\
    cout << "Registers per block: " << prop.regsPerBlock << endl;\
    cout << "Warp size: " << prop.warpSize << endl;\
    cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;\
    cout << "Max threads dim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << endl;\
    cout << "Max grid size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;\
    cout << "Const memory: " << prop.totalConstMem / 1024 << "KB" << endl;\
}

__global__ void calc_score(int jc, int j, typeof_memo *memo, typeof_memo *points)
{
    int justice = j + blockIdx.x, a = blockIdx.y, m = blockIdx.z * mul_elem(blockDim) + threadIdx.x; //mだけグリッドzとブロック内で分けてる
    size_t idx = blockIdx.x * gridDim.y * gridDim.z * mul_elem(blockDim)
        + a * gridDim.z * mul_elem(blockDim)
        + m;
    int all_notes = jc + justice + a + m;
    if (max_notes < all_notes || !all_notes) return;
    points[idx] = memo[(int)((jc * 1.01f + justice + a * 0.5f) * 1000000 / (all_notes))]
        + memo[jc] + memo[justice] + memo[a] + memo[m];
    return;
}

int main(int argc, char *argv[])
{
    get_info();

    dim3 block(32);

    //0~max_notesなので全部max_notes+1にしてる
    //mはブロック内でも複数やるからブロック内のスレッド数で除算, あまりが出たら+1
    int m_num = (max_notes+1)/mul_elem(block) + !!((max_notes+1)%mul_elem(block));
    int j_range = 1;
    //memory_usage_limitからj_rangeを決める
    while ((double)sizeof(typeof_memo) * j_range * (max_notes+1) * m_num * mul_elem(block)/1024/1024 < memory_usage_limit) j_range++;
    j_range-=1;

    dim3 grid(j_range, max_notes+1, m_num); //x: j(長さ), y: a, z: m
    typeof_memo *memo, *h_memo = new typeof_memo[1010000 + 1], *points, *h_points = new typeof_memo[mul_elem(grid) * mul_elem(block)];
    cudaDo_Check(cudaMalloc(&points, sizeof(typeof_memo) * mul_elem(grid) * mul_elem(block)));
    #pragma omp parallel for //ほんとは各桁いい感じに回せばいいけどめんどいからゴリ押す
    for (int i = 0; i <= 1010000; i++)
    {
        h_memo[i] = 0;
        string s_i = to_string(i);
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') h_memo[i]++;
    }
    cudaDo_Check(cudaMalloc(&memo, sizeof(typeof_memo) * (1010000 + 1)));
    cudaDo_Check(cudaMemcpy(memo, h_memo, sizeof(typeof_memo) * (1010000 + 1), cudaMemcpyHostToDevice));
    delete[] h_memo;

    int num_for_search = stoi(argv[1]);
    cout << "--------" << num_for_search << "--------" << endl;
    for (int jc = 0; jc <= max_notes; jc++)
    {
        for (int j = 0; j <= max_notes-jc; j+=j_range) //全部+αが探索されるから余りを考える必要はない
        {
            calc_score << <grid, block >> > (jc, j, memo, points);
            cudaDeviceSynchronize();
            kernelCheck();
            cudaDo_Check(cudaMemcpy(h_points, points, sizeof(typeof_memo) * mul_elem(grid) * mul_elem(block), cudaMemcpyDeviceToHost));
            for (int jdiff = 0; jdiff < j_range; jdiff++) for (int attack = 0; attack <= max_notes-jc-(j+jdiff); attack++)
            for (int miss = 0; miss <= max_notes-jc-(j+jdiff)-attack; miss++)
            {
                size_t idx = jdiff * grid.y * grid.z * mul_elem(block)
                    + attack * grid.z * mul_elem(block)
                    + miss;
                if (num_for_search == h_points[idx])
                    {
                        cout << jc+j+jdiff+attack+miss << " " << (int)((jc*1.01f + j +jdiff+ attack * 0.5f) * 1000000 / (jc + j + jdiff + attack + miss))
                        << " " << jc << "-" << j+jdiff << "-" << attack << "-" << miss << " " << +h_points[idx] << " 7(s)" << endl;
                    }
            }
        }
    }
    cout << "Exploration finished!" << endl;
    return 0;
}
