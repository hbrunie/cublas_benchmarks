#include <assert.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include <cublas_v2.h>
#include <thrust/complex.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>

void computeAndPrintPerf(int N, float double_msecTotal, float float_msecTotal, int nIter);
void computeAndPrintPerfMV(int N, float double_msecTotal, float float_msecTotal, int nIter);
void GPU_fill_rand(double *A, int N);
void GPU_fill_rand_complex(thrust::complex<double> *A, const int N, const int block_size);
void ConstantInitDouble(double *data, int size, double val);
void ConstantInitSimple(float *data, int size, float val);

void print_dcmatrix(const thrust::complex<double>*A, int N, bool rowmajor);
void print_fcmatrix(const thrust::complex<float>*A, int N, bool rowmajor);
void print_dmatrix(const double*A, int N, bool rowmajor);
void print_fmatrix(const float*A, int N, bool rowmajor);
