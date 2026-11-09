#include <stdio.h>

__global__
void vector_add(float *A, float *B, float *C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        C[i] = A[i] + B[i];
}


__host__
void safeCudaMalloc(void **devPtr, size_t size) {
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

int main () {
    // 1 billion floats = 4 bytes per float *1 Billion + 4GiB allocation
    // Doing this for 3 Matrices so need 12GiB of GPU memory
    int N = 1<<30;

    float* Ah = (float*)malloc(sizeof(float)*N);
    float* Bh = (float*)malloc(sizeof(float)*N);
    float* Ch = (float*)malloc(sizeof(float)*N);

    // This takes a long time
    for (int i=0; i< N; i++) {
        Ah[i] = 1;
        Bh[i] = 3;
    }

    float *Ad, *Bd, *Cd;

    // Check if enough space to allocate and error gracefully
    safeCudaMalloc((void**)&Ad, sizeof(float)*N);
    cudaMemcpy(Ad, Ah, sizeof(float)*N, cudaMemcpyHostToDevice);

    safeCudaMalloc((void**)&Bd, sizeof(float)*N);
    cudaMemcpy(Bd, Bh, sizeof(float)*N, cudaMemcpyHostToDevice);

    safeCudaMalloc((void**)&Cd, sizeof(float)*N);

    // Launch kernel
    vector_add<<<(N+511)/512,512>>>(Ad, Bd, Cd, N);

    cudaMemcpy(Ch, Cd, sizeof(float)*N, cudaMemcpyDeviceToHost);
  

    float maxError = 0.0f;
    int count = 0;
    // We then loop through all the values and verify whether they're all 4
    // This also takes a long time
    for (int i = 0; i < N; i++) {
        float error = fabsf(Ch[i] - 4.0f);
        if (error > 0)
            count++;
        maxError = max(maxError, abs(Ch[i]-4.0f));
    }
        
    printf("Max error: %f, num of errors: %d\n", maxError, count);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    free(Ah);
    free(Bh);
    free(Ch);
}