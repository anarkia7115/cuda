#include <cstdlib>
#include <iostream>
#include <vector>

#define RADIUS 3
#define BLOCK_SIZE 1024

void random_ints(std::vector<int> &a, int N)
{
        int i;
        for (i = 0; i < N; ++i)
	        a.push_back(rand() % 30);
}

#define N (2048 * 2048 * 100)
int main(void) {

  std::vector<int> a;
  random_ints(a, N);
  std::vector<int> b(RADIUS, 0);

  for (int i = RADIUS; i < N - RADIUS; i++) {
    int sum = 0;
    for (int offset = -RADIUS; offset < RADIUS; offset++) {
      sum += a.at(i + offset);
      //int num = a[i + offset];
    }
    sum += a.at(i + RADIUS);
    //int num = a[i + RADIUS];
    b.push_back(sum);
    //std::cout << num << " = " << b[i] << std::endl;
  }

  /*
  for (int i = RADIUS; i < N - RADIUS; i++) {
    for (int offset = -RADIUS; offset < RADIUS; offset++) {
      int num = a.at( i + offset );
      //int num = a[i + offset];
      std::cout << num << " + ";
    }
    int num = a.at( i + RADIUS);
    std::cout << num << " = " << b.at(i) << std::endl;
  }
  */
}
