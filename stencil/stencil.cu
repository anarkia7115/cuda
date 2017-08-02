#include <iostream>

#define RADIUS 3
#define BLOCK_SIZE 1024

__global__ void stencil_1d(int *in, int *out, int n) {

  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // check boundary
  if (gindex >= n) {
    return;
  }

  // read
  temp[lindex] = in[gindex];
  if(threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // sync
  __syncthreads();

  int result = 0;
  for(int offset = -RADIUS ; offset <= RADIUS; offset++) {
    result += temp[lindex + offset];
  }

  out[gindex] = result;
}

void random_ints(int* a, int N)
{
        int i;
        for (i = 0; i < N; ++i)
	        a[i] = rand() % 30;
}

#define N (2048 * 2048 * 100)
int main(void) {
  int *a, *b;
  int *d_a, *d_b;
  int size = N * sizeof(int);

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  a = (int *)malloc(size);
  random_ints(a, N);
  b = (int *)malloc(size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  int M = BLOCK_SIZE;
  stencil_1d<<<(N + M - 1) / M,M>>>(d_a, d_b, N);

  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  /*
  for (int i = RADIUS; i < N - RADIUS; i++) {
    for (int offset = -RADIUS; offset < RADIUS; offset++) {
      int num = a[i + offset];
      std::cout << num << " + ";
    }
    int num = a[i + RADIUS];
    std::cout << num << " = " << b[i] << std::endl;
  }
  */
}
