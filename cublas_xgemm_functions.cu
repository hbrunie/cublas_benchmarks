#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <curand.h>
#include <cublas_v2.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include "cublas_functions.h"
#include "matrix_utils.h"

/* GPU kernels */

/* Convert double into float for all matrix values */
template <int BLOCK_SIZE> __global__ void fillFloatMatrices(double *D, float *F){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    F[threadId] = (float) D[threadId];
}



/* CPU functions */
void gpu_blas_cgemm(cublasHandle_t handle, cuFloatComplex *A, cuFloatComplex *B, 
        cuFloatComplex *C, const int n){
    int lda=n,ldb=n,ldc=n;
    const cuFloatComplex alf = {(float)1., (float)1.};
    const cuFloatComplex bet = {0.,0.};
    const cuFloatComplex *alpha = &alf;
    const cuFloatComplex *beta = &bet;
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, 
            A, lda, 
            B, ldb, beta,
            C, ldc);
}
void gpu_blas_zgemm(cublasHandle_t handle,  cuDoubleComplex *A,  cuDoubleComplex *B, 
        cuDoubleComplex *C, const int n){
    int lda=n,ldb=n,ldc=n;
    const cuDoubleComplex alf = {1.,1.};
    const cuDoubleComplex bet = {0.,0.};
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, 
            reinterpret_cast< const cuDoubleComplex*>(A), lda, 
            reinterpret_cast< const cuDoubleComplex*>(B), ldb, beta,
            reinterpret_cast< cuDoubleComplex*>(C), ldc);
}

void gpu_blas_sgemm(cublasHandle_t handle, const float *A, const float *B, float *C, const int n){
    int lda=n,ldb=n,ldc=n;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
void gpu_blas_dgemm(cublasHandle_t handle, const double *A, const double *B, double *C, const int n){
    int lda=n,ldb=n,ldc=n;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* MAIN functions */

void MatrixMultiplyReal(const int N, const int block_size){
    assert(N%BSIZE == 0);
    fprintf(stderr, "Starting MatrixMultiplyReal...\n");
    // Allocate host memory for matrices A and B
    unsigned int mem_size_A = sizeof(double) * N * N;
    unsigned int mem_size_B = sizeof(double) * N * N;
    unsigned int mem_size_C = sizeof(double) * N * N;
    unsigned int mem_size_fA = sizeof(float) * N * N;
    unsigned int mem_size_fB = sizeof(float) * N * N;
    unsigned int mem_size_fC = sizeof(float) * N * N;

    double *h_A = reinterpret_cast<double *>(malloc(mem_size_A));
    double *h_B = reinterpret_cast<double *>(malloc(mem_size_B));
    double *h_C = reinterpret_cast<double *>(malloc(mem_size_C));
    float *h_fA = reinterpret_cast<float *>(malloc(mem_size_fA));
    float *h_fB = reinterpret_cast<float *>(malloc(mem_size_fB));
    float *h_fC = reinterpret_cast<float *>(malloc(mem_size_fC));

    if (h_A == NULL || h_B == NULL || h_C == NULL
            || h_fA == NULL || h_fB== NULL || h_fC == NULL) {
        fprintf(stderr, "Failed to allocate host matrix A or B or C or fA or fB or fC!\n");
        exit(EXIT_FAILURE);
    }

    // kernel iterations according to matrix size
    int nIter = 1;
    if (N>4000)
        nIter = 150;
    if (N>6000)
        nIter = 100;
    if (N>10000)
        nIter = 50;
    if (N>14000)
        nIter = 10;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    float *d_fA, *d_fB, *d_fC;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fA), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fB), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fC), mem_size_C));

    GPU_fill_rand(d_A,N);
    GPU_fill_rand(d_B,N);
    // Setup execution parameters
    const int array = N*N/block_size;
    fillFloatMatrices<BSIZE> <<< array, block_size>>>(d_A, d_fA);
    fillFloatMatrices<BSIZE> <<< array, block_size>>>(d_B, d_fB);
#ifdef DEBUG
    checkCudaErrors(cudaMemcpy(h_A, d_A, mem_size_A, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost));
    printf("A\n");
    print_dmatrix(h_A,N,true);
    printf("B\n");
    print_dmatrix(h_B,N,true);

    checkCudaErrors(cudaMemcpy(h_fA, d_fA, mem_size_fA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fB, d_fB, mem_size_fB, cudaMemcpyDeviceToHost));
    printf("fA\n");
    print_fmatrix(h_fA,N,true);
    printf("fB\n");
    print_fmatrix(h_fB,N,true);
#endif

    float double_msecTotal = 0.0f, float_msecTotal = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));

    cudaDeviceSynchronize();
    // Allocate CUDA events that we'll use for timing
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    for (int j = 0; j < nIter; j++)
        gpu_blas_dgemm(handle, d_A, d_B, d_C,N);
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&double_msecTotal, start, stop));

    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(start, NULL));
    for (int j = 0; j < nIter; j++)
        gpu_blas_sgemm(handle,d_fA,d_fB, d_fC,N);
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&float_msecTotal, start, stop));

    cublasDestroy(handle);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fC, d_fC, mem_size_fC, cudaMemcpyDeviceToHost));

    // Compute and print the performance
    computeAndPrintPerf(N, double_msecTotal, float_msecTotal, nIter);

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_fA);
    free(h_fB);
    free(h_fC);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_fA));
    checkCudaErrors(cudaFree(d_fB));
    checkCudaErrors(cudaFree(d_fC));
}
