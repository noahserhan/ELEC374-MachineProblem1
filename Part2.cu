#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>
#include <random>
#include <time.h>
#include <iostream>

__global__ void matrixMulKernel(float* P, const float* M, const float* N, int dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < dim && col < dim) {
        float sum = 0.0;
        for (int k = 0; k < dim; ++k) {
            sum += M[row * dim + k] * N[k * dim + col];
        }
        P[row * dim + col] = sum;
    }
}
