#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

void softmax_cpu(float *input, float *output, const int M, const int N)
{
    for (int m = 0; m < M; m++)
    {
        float maxval = -INFINITY;
        const float *x = input + m * N;
        for (int n = 0; n < N; n++)
        {
            maxval = maxval > x[n] ? maxval : x[n];
        }
    }
}