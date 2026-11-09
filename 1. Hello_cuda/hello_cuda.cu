#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("Hello from GPU! Global thread ID: %d, blockid: %d, blockDim: %d, threadIdx: %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
}

int main() {
    // Launch kernel with 2 blocks, 4 threads each
    helloKernel<<<2, 4>>>();
    
    // Force CPU to wait for GPU to finish and flush printf buffer
    // If you don't program will quit before GPU can flush the buffer
    cudaDeviceSynchronize();
    
    return 0;
}