#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;
#define threads_per_block 32
#define typeof_memo unsigned char
//ここでGPUメモリ(ホストメモリも)の使用量を調整
#define memory_usage_limit 1024 * 2 //MB

#define cudaDo_Check(err)\
{\
    auto tmp = err;\
    if (tmp)\
    {\
        cerr <<"!!!ERROR!!! " << endl;\
        cerr << "cudaError_t: " << tmp << " " << cudaGetErrorString(tmp) << " At " << __FILE__ << ":" << __LINE__ << endl;\
        return -1;\
    }\
}
#define kernelCheck() cudaDo_Check(cudaGetLastError())

//#define mul_elem(bl) ((bl.x + 1) * (bl.y + 1) * (bl.z + 1)) //なんか1足すとうまく動くな...?
#define mul_elem(bl) (bl.x * bl.y * bl.z) //気のせいだと思うので戻す(めもり～～のむだ)

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
}

__global__ void calc_score(int jc, int j, int max_notes, typeof_memo *memo, typeof_memo *points)
{
    int justice = j + blockIdx.x, a = blockIdx.y, m = blockIdx.z*threads_per_block + threadIdx.x; //mだけグリッドzとブロック内で分けてる
    int idx = blockIdx.x * gridDim.y * gridDim.z * threads_per_block
    + a * gridDim.z*threads_per_block
    + m;
    if (max_notes < jc + justice + a + m || !(jc + justice + a + m)) // 0除算, 無駄な計算をかいひ
    {
        points[idx] = 0;
        return;
    }
    points[idx] = memo[(int)((jc*1.01f + justice + a * 0.5f) * 1000000 / (jc + justice + a + m))]
    + memo[jc] + memo[justice] + memo[a] + memo[m];
    return;
}

int main()
{
    int max_notes = 4444;
    
    get_info();
    dim3 block(threads_per_block);


    //0~max_notesなので全部max_notes+1にしてる
    //mはブロック内でも複数やるから(max_notes+1)/threads_per_block, あまりが出たら+1
    int m_num = (max_notes+1)/threads_per_block + !!((max_notes+1)%threads_per_block);
    int j_range = 1;
    while ((double)sizeof(typeof_memo) * j_range * (max_notes+1) * m_num * mul_elem(block)/1024/1024 < memory_usage_limit) j_range++;
    j_range-=1;

    dim3 grid(j_range, max_notes+1, m_num); //x: j(長さ), y: a, z: m
    int current_max = 1, score;
    typeof_memo *memo, *host_memo = new typeof_memo[1010000 + 1], *points;
    cudaDo_Check(cudaMallocManaged(&points, sizeof(typeof_memo) * mul_elem(grid) * mul_elem(block)));
    cudaDo_Check(cudaMalloc(&memo, sizeof(typeof_memo) * (1010000 + 1))); //よく使うからこれはグローバルメモリに載せる コンスタントにはでかすぎ
    #pragma omp parallel for //ほんとは各桁いい感じに回せばいいけどめんどいからゴリ押す
    for (int i = 0; i <= 1010000; i++) {
        auto s_i = to_string(i);
        host_memo[i] = 0;
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') host_memo[i]++;
    }
    cudaDo_Check(cudaMemcpy(memo, host_memo, sizeof(typeof_memo) * (1010000 + 1), cudaMemcpyHostToDevice));
    delete[] host_memo;

    cout << "--------" << current_max << "--------" << endl;
    for (int jc = 0; jc <= max_notes; jc++)
    {
        for (int j = 0; j <= max_notes-jc; j+=j_range)
        {
            //cout << "Running calc_score kernel with jc=" << jc << ", j=" << j<< "~" << j+j_range-1 << endl;
            calc_score << <grid, block >> > (jc, j, max_notes, memo, points);
            cudaDeviceSynchronize();
            kernelCheck();
            for (int jdiff = 0; jdiff < j_range; jdiff++) for (int attack = 0; attack <= max_notes-jc-j; attack++)
            for (int miss = 0; miss <= max_notes-jc-j-attack; miss++)
            {
                int idx = jdiff * (max_notes + 1) * m_num * threads_per_block + attack * m_num * 
                    threads_per_block + miss;
                    if (current_max <= points[idx]) {
                        if(current_max < points[idx])
                        {
                            current_max = points[idx];
                            cout << "--------" << current_max << "--------" << endl;
                        }
                        score = (jc*1.01f + j +jdiff+ attack * 0.5f) * 1000000 / (jc + j + jdiff + attack + miss);
                        cout << jc+j+jdiff+attack+miss << " " << score << " "
                        << jc << "-" << j+jdiff << "-" << attack << "-" << miss << " " << +points[idx] << " 7(s)" << endl;
                    }
            }
        }

        // 残ったjusticeを処理
        grid.x = 1;
        for (int j = (max_notes - jc + 1) / j_range * j_range; j <= max_notes-jc; j++)
        {
            calc_score << <grid, block >> > (jc, j, max_notes, memo, points);
            cudaDeviceSynchronize();
            kernelCheck();
            for (int attack = 0; attack <= max_notes-jc-j; attack++) {
                for (int miss = 0; miss <= max_notes-jc-j-attack; miss++)
                {
                    int idx = attack * m_num * threads_per_block + miss;
                    if (current_max <= points[idx]) {
                        if(current_max < points[idx])
                        {
                            current_max = points[idx];
                            cout << "--------" << current_max << "--------" << endl;
                        }
                        score = (jc*1.01f + j + attack * 0.5f) * 1000000 / (jc + j + attack + miss);
                        cout << jc+j+attack+miss << " " << score << " "
                        << jc << "-" << j << "-" << attack << "-" << miss << " " << +points[idx] << " 7(s)" << endl;
                    }
                }
            }
        }
        grid.x = j_range;
    }

    cout << "Exploration finished!" << endl;
    return 0;
}
