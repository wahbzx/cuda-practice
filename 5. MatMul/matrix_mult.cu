#include <stdio.h>
#include <stdlib.h>

void cpuMatMul(float *M, float *N, float *P, int width) {
    for (int i =0; i < width; i++) {
        for (int j=0; j< width; j++) {
            float sum = 0;
            for (int k =0; k < width; k++) {
                float a = M[i * width + k];
                float b = N[k * width + j];
                sum += a*b;
            }
            P[i * width + j] = sum;
        }
    }
}

__global__ void matrix_mul_kernel(float *M, float *N, float *P, int width) {
    int tx = threadIdx.x; // Identifies the row of Md
    int ty = threadIdx.y; // Identifies the column of Nd

    // Each thread will be responsible for calculating 1 value of p
    float Pvalue = 0;
    for (int k =0; k < width; k++) {
            float a = M[ty * width + k];
            float b = N[k * width + tx];
            Pvalue += a * b;
        }
    P[ty * width + tx] = Pvalue;
    
}

void matrix_mul(float *M, float *N, float *P, int width) {
    int size = (width * width * sizeof(float));
    float *Md, *Nd, *Pd;
    cudaMalloc((void**)&Md, size); 
    cudaMemcpy(Md, M, sizeof(M), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Nd, size); 
    cudaMemcpy(Nd, N, sizeof(N), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&Pd, size);     

    // Launch kernel
    dim3 block_dim (width, width);
    dim3 grid_dim (1, 1);

    matrix_mul_kernel<<<grid_dim, block_dim>>>(Md, Nd, Pd, width);

    // Transfer the result back to P from the device
    cudaMemcpy(P, Pd, sizeof(P), cudaMemcpyDeviceToHost);

    // Clean up allocated device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

int main () {
    int width = 100;
    int size = (width * width * sizeof(float));
    // allocate memory on host
    float *M = (float *)malloc(size);
    float *N = (float *)malloc(size);
    float *P = (float *)malloc(size);

    // Do the multiplication however
    matrix_mul(M, N, P, width);


    free(M);
    free(N);
    free(P);

}