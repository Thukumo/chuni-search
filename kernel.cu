#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
using namespace std;

__global__ void calc_score(int jc, int j, int a, int range_calc, int *memo, int *score, int *points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (range_calc < idx) return; //idx <= range_calcでfor
    //*score(int)で切り捨て
    score[idx] = (jc*1.01f + j + a * 0.5f) * 1000000 / (jc + j + a + idx); //0-0-0-0のときはここでコケるから、jc, j, aのループを逆順で回すと0-0-0-0の場合がおかしくなるはず
    points[idx] = memo[score[idx]] + memo[jc] + memo[j] + memo[a] + memo[idx]; //ランダムアクセスだからGPUに投げた方が遅い説ない?
    return;
}

int main()
{
    int max_notes = 4444;

    dim3 block(256);
    dim3 grid((max_notes + block.x - 1) / block.x);
    int *score, *points, current_max = 1, *memo;
    cudaMallocManaged(&points, sizeof(int) * (max_notes+1));
    cudaMallocManaged(&memo, sizeof(int) * (1010000+1));
    //ほんとは各桁いい感じに回せばいいけどめんどいからゴリ押す
    #pragma omp parallel for
    for (int i = 0; i <= 1010000; i++) {
        auto s_i = to_string(i);
        for (int j = 0; j < s_i.size(); j++) if (s_i[j] == '7') memo[i]++;
    }
    cudaMallocManaged(&score, sizeof(int) * (max_notes+1));
    cout << "--------" << current_max << "--------" << endl;

    for (int jc = 0; jc <= max_notes; jc++) {
    //for (int jc = max_notes; 0 <= jc; jc--) {
        for (int j = 0; j <= max_notes - jc; j++)
        {
            for (int a = 0; a <= max_notes - jc - j; a++)
            {
                calc_score<<<grid, block>>>(jc, j, a, max_notes-jc-j-a, memo, score, points);
                cudaDeviceSynchronize();
                for (int i = 0; i <= max_notes - jc - j - a; i++) if (current_max <= points[i])
                {
                        if (current_max < points[i])
                        {
                            current_max = points[i];
                            cout << "--------" << current_max << "--------" << endl;
                        }
                        cout << jc + j + a + i << " " << score[i] << " " << jc << "-" << j << "-" << a << "-" << i << " " << current_max << " 7(s)" << endl;
                }
            }
        }
    }
    cout << "Exploration finished!" << endl;
    return 0;
}
