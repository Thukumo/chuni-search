#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;
#define threads_per_block 32 //真に一度の実行で終わらせられそうだからwarpに合わせる
#define typeof_memo uint8_t

__global__ void calc_score(int jc, int max_notes, typeof_memo *memo, int *score, typeof_memo *points) {
    int j = blockIdx.x, a = blockIdx.y, m = blockIdx.z*threads_per_block + threadIdx.x; //mだけグリッドzとブロック内で分けてる
    int idx = j * max_notes * max_notes + a * max_notes + m;
    if (max_notes < jc + j + a + m) return;
    else if (jc + j + a + m)
    {
        score[idx] = 0;
        points[idx] = 0;
        return;
    }
    //intへのキャストで切り捨て
    score[idx] = (jc*1.01f + j + a * 0.5f) * 1000000 / (jc + j + a + m); //0-0-0-0のときはここでコケるから、jc, j, aのループを逆順で回すと0-0-0-0の場合がおかしくなるはず
    points[idx] = memo[score[idx]] + memo[jc] + memo[j] + memo[a] + memo[m];
    return;
}

int main()
{
    int max_notes = 4444;

    dim3 block(threads_per_block);
    //0~max_notesなので全部max_notes+1となってる
    //mはブロック内でも複数やるから(max_notes+1)/threads_per_block, あまりが出たら+1してる
    int m = (max_notes+1)/threads_per_block + !(int)((double)(max_notes+1)/threads_per_block - (max_notes+1)/threads_per_block);
    dim3 grid(max_notes+1, max_notes+1, m); //x: j, y: a, z: m
    int *score, current_max = 1;
    typeof_memo *memo, *host_memo = new typeof_memo[1010000 + 1], *points;
    cudaMallocManaged(&points, sizeof(typeof_memo) * (max_notes+1 * max_notes+1 * max_notes+1));

    cudaMalloc(&memo, sizeof(typeof_memo) * (1010000 + 1)); //よく使うからこれはグローバルメモリに載せる
    #pragma omp parallel for //ほんとは各桁いい感じに回せばいいけどめんどいからゴリ押す
    for (int i = 0; i <= 1010000; i++) {
        auto s_i = to_string(i);
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') host_memo[i]++;
    }
    cudaMemcpy(memo, host_memo, sizeof(typeof_memo) * (1010000 + 1), cudaMemcpyHostToDevice);
    delete[] host_memo;

    cudaMallocManaged(&score, sizeof(int) * (max_notes * max_notes * max_notes));
    cout << "--------" << current_max << "--------" << endl;

    for (int jc = 0; jc <= max_notes; jc++) {
        calc_score << <grid, block >> > (jc, max_notes, memo, score, points);
        cudaDeviceSynchronize();
        for (int j = 0; j <= max_notes; j++) {
            for (int a = 0; a <= max_notes; a++) {
                for (int m = 0; m <= max_notes; m++) {
                    int idx = j * max_notes * max_notes + a * max_notes + m;
                    if (current_max <= points[idx]) {
                        if(current_max < points[idx])
                        {
                            current_max = points[idx];
                            cout << "--------" << current_max << "--------" << endl;
                        }
                        cout << jc+j+a+m << " " << score[idx] << " "
                        << jc << "-" << j << "-" << a << "-" << m << " " << points[idx] << "7(s)" << endl;
                    }
                }
            }
        }
    }
    cout << "Exploration finished!" << endl;
    return 0;
}
