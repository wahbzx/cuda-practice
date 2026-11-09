#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(void)
{
    int N = 1<<20;
    float *x, *y, *z;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    z = (float*)malloc(N*sizeof(float));

    

    // To take into account the memory access similar to the cuda example
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // For high precision time use timespec
    struct timespec start, end;
    // Get the starting timestamp
    timespec_get(&start, TIME_UTC);
    // Record the starting clock tick

    for (int i = 0; i < N; i++) {
        // Perform SAXPY
        z[i] = 2.0f*x[i] + y[i];
        // z[i] = 2.0f*1.0f + 2.0f;
    }

    
    timespec_get(&end, TIME_UTC);
    // Calculate total time in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("SAXPY on %d values took %f ms to execute.\n", N, time_taken);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, abs(z[i]-4.0f));
    printf("Max error: %f\n", maxError);

    free(x);
    free(y);
    free(z);

}