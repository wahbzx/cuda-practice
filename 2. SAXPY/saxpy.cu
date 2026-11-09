#include <stdio.h>

// https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
// CUDA kernel to perform "Single-Precision A·X Plus Y" SAXPY a hello world equivelant in the world of parallel computing
// Use this keyword to indicate its a kernel to be ran on the GPU
__global__
void saxpy(int n, float a, float *x, float *y)
{
  // Because threads are broken up into blocks, a thread needs to calculate its absolute global position in the grid to index an array safely. 
  // For a simple 1D grid, the formula is:
  // In simple terms: We find how many blocks into the grid we are, then how far in the block we are
  // E.g. First block, 4 thread would be 0 * blockDim + 4
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  // a * x + y = 2*1 + 2 - So we expect y to be filled with 4s after
  // For cases where the number of elements in the arrays is not evenly divisible by the thread block size, 
  // the kernel code must check for out-of-bounds memory accesses.

  if (i < n) y[i] = a*x[i] + y[i];
}

// Host code - Run on the CPU
int main(void)
{
  // Initialise array size of 2^20 = 1,048,576
  int N = 1<<20;
  printf("Preparing %i values\n", N);
  // Declare 2 pairs of  arrays
  // NOTE: The host and device in CUDA have separate memory spaces,
  float *x, *y, *d_x, *d_y;

  // Host arrays
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  // Device Arrays 
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  //The host code then initializes the host arrays.
  // Here we set x to an array of ones, and y to an array of twos.
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // We then copy the values in the host arrays to the device arrays
  // Works similar to regular memcpy but takes a 4th argument for the direction
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements - Launching our CUDA Kernel with a grid of thread blocks
  // When executing a kernel it takes an execution configuration inbetween the triple chevrons 
  // <<< Grid size ( blocks in the grid), Block size (threads in the block) >>>
  // We add 255 to force the integer division to round up to the next whole block in case its not perfectly divisble
  // Then in brackets its just the arguments to the kernel

  // This example: Grid size = 4,096 = (1048576 + 255) / 256, Block size = 256
  // Total threads = grid size * block size = 4,096 * 256 = 1,048,576
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);


  // After performing our operation we copy the result back to the host array y
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  // We then loop through all the values and verify whether they're all 4
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  


  // Clean up, need to free both GPU memory and Main memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}