#include <iostream>
using namespace std;

//こいつらを元に進捗を表示するようにさせたい
#define overl endl;
#define up(n) cout << "\033[" << n << "F"
#define down(n) cout << "\033[" << n << "E"
#define line_clear() cout << "\033[2K"
#define endl endl; endlCount++
#define reset() endlCount = 0

int endlCount = 0;

int main() {
    cout << "Hello, World!" << endl;
    _sleep(1000);
    reset();
    cout << "---" << 1 << "---" << endl;
    for (int i = 2; i < 10; i++) {
        up(endlCount);
        line_clear();
        reset();
        cout << "---" << i << "---" << endl;
        _sleep(1000);
    }
    return 0;
}
