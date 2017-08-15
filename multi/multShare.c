#include "multShare.h"

void MatMul(const Matrix A, const Matrix B, Matrix C) {
  // Load A and B to device
  Matrix d_A;
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaError_t err = cudaMalloc(&d_A.elements, size);
  printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Matrix d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  err = cudaMalloc(&d_B.elements, size);
  printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  // Allocate C in device
  Matrix d_C;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  err = cudaMalloc(&d_C.elements, size);
  printf("CUDA malloc C: %s\n", cudaGetErrorString(err));

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  int gridx = (B.width + dimBlock.x - 1) / dimBlock.x;
  int gridy = (A.height + dimBlock.y - 1) / dimBlock.y;

  dim3 dimGrid(gridx, gridy);
  printf("dimGrid: %d, %d, dimBlock: %d, %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  //cudaDeviceSynchronize();
  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory
  err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  printf("Copy C off of device: %s\n", cudaGetErrorString(err));

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
  float rst(0);
  if (row < A.height && col < A.width){
    rst = A.elements[row * A.stride + col];
  }
  return rst;
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {

  if (row < A.height && col < A.width){
    //printf("%d, %d exceed %d, %d", row, A.height, 
    //    col, A.width);
    //value = 0;
    A.elements[row * A.stride + col] = value;
  }
  //printf("Set %d, %d to %f\n", row, col, value);
}

// Get the BLOCK_SIZE x BLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int blockRow, int blockCol) {

  Matrix Asub;

  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;

  // limit Asub height;
  blockRow = blockRow + 1;
  if (blockRow * BLOCK_SIZE > A.height) {
    int diff = blockRow * BLOCK_SIZE - A.height;
    Asub.height = Asub.height - diff;
  } else if (blockRow * BLOCK_SIZE == A.height) {
    //printf("not gonna happen!");
  }

  // limit Asub width;
  blockCol = blockCol + 1;
  if (blockCol * BLOCK_SIZE > A.width) {
    int diff = blockCol * BLOCK_SIZE - A.width;
    Asub.width = Asub.width - diff;
  } else if (blockCol * BLOCK_SIZE == A.width) {
    //printf("not gonna happen!");
  }

  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * (blockRow - 1) + BLOCK_SIZE * (blockCol - 1)];
  //printf("block_size: %d\n", BLOCK_SIZE);
  //printf("height: %d\twidth: %d\t at (%d,%d)\n"
  //    , Asub.height, Asub.width, blockRow, blockCol);
  return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  //printf("%d, %d, %d, %d\n", blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y);
  //printf("%d,%d\n", blockIdx.x * BLOCK_SIZE + threadIdx.x, blockIdx.y * BLOCK_SIZE + threadIdx.y);


  // Each thread block computes one sub-matrix Csub of C
  //printf("get C\n");
  //printf("blockRow: %d, blockCol: %d\n", blockRow, blockCol);
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0.0;

  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  //printf("A.width: %d", A.width);
  for (int m = 0; m  < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
    // Get sub-matrix Asub of A
    //printf("get A");
    Matrix Asub = GetSubMatrix(A, blockRow, m);

    // Get sub-matrix Bsub of B
    //printf("get B");
    Matrix Bsub = GetSubMatrix(B, m, blockCol);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
    //printf("A(x, y): %d, %d\t%f\n", row, col, As[row][col]);
    //printf("B(x, y): %d, %d\t%f\n", row, col, Bs[row][col]);

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      //printf("in for loop");
      Cvalue += As[row][e] * Bs[e][col];
    }

    // Synchronize to make sure that the preceding 
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
}


int main(int argc, char* argv[]) {
  Matrix A, B, C;
  int a1, a2, b1, b2, rep_size;
  a1 = atoi(argv[1]);
  a2 = atoi(argv[2]);
  b1 = a2;

  b2 = atoi(argv[3]);
  rep_size = atoi(argv[4]);

  A.height = a1;
  A.width = a2;
  A.elements = (float*)malloc(A.width * A.height * sizeof(float));

  B.height = b1;
  B.width = b2;
  B.elements = (float*)malloc(B.width * B.height * sizeof(float));

  C.height = A.height;
  C.width = B.width;
  C.elements = (float*)malloc(C.width * C.height * sizeof(float));

  for(int i = 0; i < A.height; i++) {
    for(int j = 0; j < A.width; j++) {
      A.elements[i*A.width + j] = (float)(rand() % 3);
    }
  }

  for(int i = 0; i < B.height; i++) {
    for(int j = 0; j < B.width; j++) {
      B.elements[i*B.width + j] = (float)(rand() % 2);
    }
  }

  printf("repeat %d times. \n", rep_size);
  for (int i = 0; i < rep_size; i++) {
    MatMul(A, B, C);
  }

  //print
  for(int i = 0; i < min(20, A.height); i++) {
    for(int j = 0; j < min(20, A.width); j++) {
      printf("%f ", A.elements[i * A.width + j]);
    }
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(20, B.height); i++) {
    for(int j = 0; j < min(20, B.width); j++) {
      printf("%f ", B.elements[i * B.width + j]);
    }
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(20, C.height); i++) {
    for(int j = 0; j < min(20, C.width); j++) {
      printf("%f ", C.elements[i * C.width + j]);
    }
    printf("\n");
  }
  printf("\n");

  // sum avg
  float c_sum = 0;

  for(int i = 0; i < C.height; i++) {
    for(int j = 0; j < C.width; j++) {
      c_sum += C.elements[i * C.width + j];
    }
  }

  float c_avg = c_sum / (C.height * C.width);
  printf("avg: %f", c_avg);
}
