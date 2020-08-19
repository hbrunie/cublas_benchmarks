#include "cublas_functions.h"
#include "matrix_utils.h"

/* GPU kernels */

/* Convert double into float for all matrix values */
template <int BLOCK_SIZE> __global__ void fillFloatMatrix(double *D, float *F){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    F[threadId] = (float) D[threadId];
}

void gpu_blas_sgemv(cublasHandle_t handle, const float *A, const float *X, float *Y, const int n){
    int lda=n,incx=1,incy=1;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasSgemv(handle, CUBLAS_OP_N,
                n, n, alpha, A, lda, X, incx, beta, Y, incy);
}

void gpu_blas_dgemv(cublasHandle_t handle, const double *A, const double *X, double *Y, const int n){
    int lda=n,incx=1,incy=1;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasDgemv(handle, CUBLAS_OP_N,
                n, n, alpha, A, lda, X, incx, beta, Y, incy);
}

/* MAIN functions */

void MatrixVectorMultiplyReal(const int N, const int block_size){
    assert(N%block_size == 0);
    // Allocate host memory for matrices A and vector B
    unsigned int mem_size_A = sizeof(double) * N * N;
    unsigned int mem_size_X = sizeof(double) * N;
    unsigned int mem_size_Y = sizeof(double) * N * N;
    unsigned int mem_size_fA = sizeof(float) * N * N;
    unsigned int mem_size_fX = sizeof(float) * N;
    unsigned int mem_size_fY = sizeof(float) * N * N;

    double *h_A = reinterpret_cast<double *>(malloc(mem_size_A));
    double *h_X = reinterpret_cast<double *>(malloc(mem_size_X));
    double *h_Y = reinterpret_cast<double *>(malloc(mem_size_Y));
    float *h_fA = reinterpret_cast<float *>(malloc(mem_size_fA));
    float *h_fX = reinterpret_cast<float *>(malloc(mem_size_fX));
    float *h_fY = reinterpret_cast<float *>(malloc(mem_size_fY));

    if (h_A == NULL || h_X == NULL || h_Y == NULL
            || h_fA == NULL || h_fX== NULL || h_fY == NULL) {
        fprintf(stderr, "Failed to allocate host matrix A or X or Y or fA or fX or fY!\n");
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
    double *d_A, *d_X, *d_Y;
    float *d_fA, *d_fX, *d_fY;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_X), mem_size_X));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Y), mem_size_Y));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fA), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fX), mem_size_X));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_fY), mem_size_Y));

    GPU_fill_rand(d_A,N*N);
    GPU_fill_rand(d_X,N);
    // Setup execution parameters
    fillFloatMatrix<BSIZE> <<<N*N/block_size, block_size>>>(d_A, d_fA);
    fillFloatMatrix<BSIZE> <<< N/block_size, block_size>>>(d_X, d_fX);
#ifdef DEBUG
    checkCudaErrors(cudaMemcpy(h_A, d_A, mem_size_A, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_X, d_X, mem_size_X, cudaMemcpyDeviceToHost));
    printf("A\n");
    print_dmatrix(h_A,N,true);
    printf("X\n");
    print_dmatrix(h_X,N,true);

    checkCudaErrors(cudaMemcpy(h_fA, d_fA, mem_size_fA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fX, d_fX, mem_size_fX, cudaMemcpyDeviceToHost));
    printf("fA\n");
    print_fmatrix(h_fA,N,true);
    printf("fX\n");
    print_fmatrix(h_fX,N,true);
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
        gpu_blas_dgemv(handle, d_A, d_X, d_Y,N);
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&double_msecTotal, start, stop));

    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(start, NULL));
    for (int j = 0; j < nIter; j++)
        gpu_blas_sgemv(handle,d_fA,d_fX, d_fY,N);
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&float_msecTotal, start, stop));

    cudaDeviceSynchronize();

    cublasDestroy(handle);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fY, d_fY, mem_size_fY, cudaMemcpyDeviceToHost));

    // Compute and print the performance
    computeAndPrintPerfMV(N, double_msecTotal, float_msecTotal, nIter);

    // Clean up memory
    free(h_A);
    free(h_X);
    free(h_Y);
    free(h_fA);
    free(h_fX);
    free(h_fY);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Y));
    checkCudaErrors(cudaFree(d_fA));
    checkCudaErrors(cudaFree(d_fX));
    checkCudaErrors(cudaFree(d_fY));
}
