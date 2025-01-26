#include <iostream>

using namespace std;
int main() {
    int jc = 5, j = 2, attack = 0, miss = 0;
    int score = (jc*1.01f + j + attack * 0.5f) * 1000000 / (jc + j + attack + miss);
    cout << "Score: " << score << endl;
    return 0;
}
