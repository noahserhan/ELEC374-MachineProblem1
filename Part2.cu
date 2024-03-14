// Noah Serhan - 20302832
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <random>
#include <time.h>
#include <iostream>

const int matrixSizes[] = { 100, 250, 500, 1000, 1500 };
const int blockWidths[] = { 2, 5, 10, 25, 32 };


// Random float generation for matrix
float randomFloat() {
    float coefficient = (float)rand() / (float)RAND_MAX;
    return  coefficient * 20.0 - 10.0;  // MIN -> -10.0, MAX -> 10.0 
}

// Filling in the matrix with the random floats
void buildMatrix(float* matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrix[dim * i + j] = randomFloat();
        }
    }
}

__global__ void matrixMulKernel(float* P, const float* M, const float* N, int dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < dim && col < dim) {
        float sum = 0.0;
        for (int x = 0; x < dim; x++) {
            sum += M[row * dim + x] * N[x * dim + col];
        }
        P[row * dim + col] = sum;
    }
}

// For matrix mult for the single thread
__global__ void singleThreadMatrixMul(float* P, const float* M, const float* N, int dim) {
    for (int i = 0; i < dim * dim; i++) {
        P[i] = 0;
        int startOfRow = i - i % dim;
        int startOfColumn = i % dim;
        for (int j = 0; j < dim; j++) {
            P[i] += M[startOfRow + j] * N[startOfColumn + j * dim];
        }
    }
}

__global__ void matrixMulHost(float* P, const float* M, const float* N, int dim) {
    for (int i = 0; i < dim * dim; i++) {
        P[i] = 0;
        int startOfRow = i - i % dim;
        int startOfColumn = i % dim;
        for (int j = 0; j < dim; j++) {
            P[i] += M[startOfRow + j] * N[startOfColumn + j * dim]; 
        }
    }
}

int compareMatrices(float* matrix1, float* matrix2, int dim) {
    float tolerance = 0.1f;
    for (int i = 0; i < dim * dim; i++) {
        float diff = matrix1[i] - matrix2[i];
        if (diff > tolerance || diff < -tolerance) {
            float ratio = matrix1[1] / matrix2[i];
            if (ratio > 1.0f + tolerance || ratio < 1.0f - tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    int numOfMatrices = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int numOfBlockWidths = sizeof(blockWidths) / sizeof(blockWidths[0]);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    cudaEvent_t startTransferToDev, stopTransferToDev;
    cudaEventCreate(&startTransferToDev);
    cudaEventCreate(&stopTransferToDev);

    cudaEvent_t startTransferToHost, stopTransferToHost;
    cudaEventCreate(&startTransferToHost);
    cudaEventCreate(&stopTransferToHost);

    cudaEvent_t startDevMult, stopDevMult;
    cudaEventCreate(&startDevMult);
    cudaEventCreate(&stopDevMult);

    cudaEvent_t startKernelMult, stopKernelMult;
    cudaEventCreate(&startKernelMult);
    cudaEventCreate(&stopKernelMult);

    cudaEvent_t startHostMult, stopHostMult;
    cudaEventCreate(&startHostMult);
    cudaEventCreate(&stopHostMult);

    cudaEvent_t startSingleTheadMult, stopSingleThreadMult;
    cudaEventCreate(&startSingleTheadMult);
    cudaEventCreate(&stopSingleThreadMult);
    
    for (int i = 0; i < numOfMatrices; i++) {
        size_t size = matrixSizes[i];
        size_t trueSize = size * size * sizeof(float);

        std::cout << "\n\n\nMatrix" << i + 1 << " with a dimension of" << size << " being tested." << std::endl;

        float* M;
        float* N;
        float* P;
        float* cpuP;
        cudaMallocHost(&M, trueSize);
        cudaMallocHost(&N, trueSize);
        cudaMallocHost(&P, trueSize);
        cudaMallocHost(&cpuP, trueSize);

        float* d_M;
        float* d_N;
        float* d_P;
        cudaMallocHost(&d_M, trueSize);
        cudaMallocHost(&d_N, trueSize);
        cudaMallocHost(&d_P, trueSize);

    
        buildMatrix(M, size);
        buildMatrix(N, size);

        float transferTime = 0.0f;

        std::cout << "Beginning transfer from host to device." << std::endl;
        cudaDeviceSynchronize();
        cudaEventRecord(startTransferToDev, 0);
        cudaMemcpyAsync(d_M, M, trueSize, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_N, N, trueSize, cudaMemcpyHostToDevice);
        cudaEventRecord(stopTransferToDev, 0);
        cudaEventSynchronize(stopTransferToDev);
        cudaEventElapsedTime(&transferTime, startTransferToDev, stopTransferToDev);
        std::cout << "Transfer complete after " << transferTime << " ms." << std::endl;

        // COMMENT OUT FOR SINGLE THREAD GPU MULTIPLICATION
        //std::cout << "Beginning single thread device multiplication." << std::endl;
        //cudaDeviceSynchronize();
        //float singleThreadTime = 0.0f;
        //cudaEventRecord(startSingleTheadMult);
        //matrixMulKernel <<<1, 1>>> (size, d_P, d_M, d_N); // Expected red squiggly line - IGNORE
        //cudaEventRecord(stopSingleThreadMult);
        //cudaEventSynchronize(stopSingleThreadMult);
        //cudaEventElapsedTime(&singleThreadTime, startSingleTheadMult, stopSingleThreadMult);
        //std::cout << "Operation complete after " << singleThreadTime << " ms." << std::endl;

        // COMMENT OUT FOR SINGLE MATRIX ENTRY MULTIPLICATION
        //std::cout << "Beginning single entry device multiplication." << std::endl;
        //cudaDeviceSynchronize();
        //float singleEntryTime = 0.0f;
        //cudaEventRecord(startKernelMult);
        //matrixMulKernel <<<1, 1>>> (size, d_P, d_M, d_N); // Expected red squiggly line - IGNORE
        //cudaEventRecord(stopKernelMult);
        //cudaEventSynchronize(stopKernelMult);
        //cudaEventElapsedTime(&singleEntryTime, startKernelMult, stopKernelMult);
        //std::cout << "Operation complete after " << singleEntryTime << " ms." << std::endl;


        for (int i = 0; i < numOfBlockWidths; i++) {
            std::cout << "Beginning multiplication with block width: " << blockWidths[i] << std::endl;
            
            float multTime = 0.0f;

            int threadDim = blockWidths[i];
            dim3 threads = dim3(threadDim, threadDim);
            int blockDim = size / threadDim;
            if (size % threadDim) blockDim++;
            dim3 blocks = dim3(blockDim, blockDim);

            cudaDeviceSynchronize();
            cudaEventRecord(startDevMult);
            matrixMulKernel <<<blocks, threads, 0, 0 >>> (dim, d_P, d_M, d_N); // IGNORE RED SQUIGGLY
            cudaEventRecord(stopDevMult);
            cudaEventSynchronize(stopDevMult);
            cudaEventElapsedTime(&multTime, startDevMult, stopDevMult);
            std::cout << "Operation complete after " << multTime << " ms.\n" << std::endl;
        }

        // HOST MULT
        std::cout << "Beginning single threaded multiplication on host" << std::endl;
        cudaDeviceSynchronize();
        float hostTime = 0.0f;
        cudaEventRecord(startHostMult);
        matrixMulHost(cpuP, M, N, size);
        cudaEventRecord(stopHostMult);
        cudaEventSynchronize(stopHostMult);
        cudaEventElapsedTime(&hostTime, startHostMult, stopHostMult);
        std::cout << "Operation complete after " << hostTime << " milliseconds." << std::endl;

        if (compareMatrices(P, cpuP, size)) {
            std::cout << "\nTEST PASSED\n" << std::endl;
        }

        cudaFreeHost(&M);
        cudaFreeHost(&N);
        cudaFreeHost(&P);
        cudaFreeHost(&cpuP);

        cudaFree(&d_M);
        cudaFree(&d_N);
        cudaFree(&d_P);
        std::cout << std::endl;
    }
    return 0;
}
