#include <iostream>

using namespace std;

int endl_count = 0;

std::ostream& my_endl(std::ostream& os) {
    ++endl_count;
    return os << std::endl;
}

int main() {
    
    return 0;
}
