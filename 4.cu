#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;
#define threads_per_block 32 //真に一度の実行で終わらせられそうだからwarpに合わせる
#define typeof_memo short

__global__ void calc_score(int jc, int j, int max_notes, typeof_memo *memo, typeof_memo *points) {
    int justice = j + blockIdx.x, a = blockIdx.y, m = blockIdx.z*threads_per_block + threadIdx.x; //mだけグリッドzとブロック内で分けてる
    int idx = blockIdx.x * threads_per_block*blockDim.z * (max_notes + 1) + a * blockDim.z*threads_per_block + m;
    if (max_notes < jc + justice + a + m || !(jc + justice + a + m))
    {
        points[idx] = 0;
        return;
    }
    points[idx] = memo[(size_t)((jc*1.01f + justice + a * 0.5f) * 1000000 / (jc + justice + a + m))] + memo[jc] + memo[justice] + memo[a] + memo[m];
    return;
}

int main()
{
    int max_notes = 4444;
    dim3 block(threads_per_block);
    int j_range = 1;
    while ((double)sizeof(typeof_memo) * (max_notes+1) * j_range * (max_notes+1)/1024/1024 < 1024 * 4) j_range++;
    j_range-=1;
    //0~max_notesなので全部max_notes+1となってる
    //mはブロック内でも複数やるから(max_notes+1)/threads_per_block, あまりが出たら+1してる
    int m_num = (max_notes+1)/threads_per_block + !((max_notes+1)%threads_per_block);

    dim3 grid(j_range, max_notes+1, m_num); //x: a, y: 未使用, z: m
    int current_max = 1, score;
    typeof_memo *memo, *host_memo = new typeof_memo[1010000 + 1], *points;
    cudaMallocManaged(&points, sizeof(typeof_memo) *grid.x * grid.y * grid.z * threads_per_block);

    cudaMalloc(&memo, sizeof(typeof_memo) * (1010000 + 1)); //よく使うからこれはグローバルメモリに載せる
    #pragma omp parallel for //ほんとは各桁いい感じに回せばいいけどめんどいからゴリ押す
    for (int i = 0; i <= 1010000; i++) {
        auto s_i = to_string(i);
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') host_memo[i]++;
    }
    cudaMemcpy(memo, host_memo, sizeof(typeof_memo) * (1010000 + 1), cudaMemcpyHostToDevice);
    delete[] host_memo;

    cout << "--------" << current_max << "--------" << endl;
    for (int jc = 0; jc <= max_notes; jc++)
    {
        for (int j = 0; j <= max_notes-jc; j+=j_range)
        {
            calc_score << <grid, block >> > (jc, j, max_notes, memo, points);
            cudaDeviceSynchronize();
            for (int jdiff = 0; jdiff < j_range; jdiff++)
            for (int attack = 0; attack <= max_notes-jc-j; attack++) {
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
                        << jc << "-" << j+jdiff << "-" << attack << "-" << miss << " " << points[idx] << " 7(s)" << endl;
                    }
                }
            }
        }

        // 残ったjusticeを処理
        grid.x = 1;
        for (int j = (max_notes - jc + 1) / j_range * j_range; j <= max_notes-jc; j++)
        {
            calc_score << <grid, block >> > (jc, j, max_notes, memo, points);
            cudaDeviceSynchronize();
            
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
                        << jc << "-" << j << "-" << attack << "-" << miss << " " << points[idx] << " 7(s)" << endl;
                    }
                }
            }
        }
        grid.x = j_range;
    }

    cout << "Exploration finished!" << endl;
    return 0;
}
