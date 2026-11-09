#include <stdio.h>
#include <cuda_runtime.h> // For cuda events

__global__
void saxpy(int n, float a, float *x, float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = a*x[i] + y[i];
}

int main(void)
{   
    // Create CUDA events to measure time correctly
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 1<<20;
    float *x, *y, *z, *d_x, *d_y, *d_z;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    z = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }

    cudaEventRecord(start, 0);

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time between two events
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cudaMemcpy execution time: %f ms\n", milliseconds);

    // Warm up the GPU by running the kernels a few times, avoids all the initial overhead.
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y, d_z);
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y, d_z);

    // Record the start event in the default stream
    cudaEventRecord(start, 0);

    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y, d_z);

    // Record the stop event
    cudaEventRecord(stop, 0);
    // Wait for the stop event to complete on the GPU
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);


    cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(z[i]-4.0f));
    printf("Max error: %f\n", maxError);


    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(x);
    free(y);
    free(z);
}