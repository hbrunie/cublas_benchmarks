#include "matrix_utils.h"
/* Compute and print results */
void computeAndPrintPerfMV(int N, float double_msecTotal, float float_msecTotal, int nIter){
    float msecPerDoubleMatrixMul = double_msecTotal / nIter;
    float msecPerFloatMatrixMul = float_msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(N) *
        static_cast<double>(N);
    double doubleGigaFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerDoubleMatrixMul / 1000.0f);
    double floatGigaFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerFloatMatrixMul / 1000.0f);
    fprintf(stderr,
            "%d %.2f %.3f %.2f %.3f\n",
            N,
            floatGigaFlops,
            msecPerFloatMatrixMul,
            doubleGigaFlops,
            msecPerDoubleMatrixMul);
}
void computeAndPrintPerf(int N, float double_msecTotal, float float_msecTotal, int nIter){
    float msecPerDoubleMatrixMul = double_msecTotal / nIter;
    float msecPerFloatMatrixMul = float_msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(N) *
        static_cast<double>(N) *
        static_cast<double>(N);
    double doubleGigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerDoubleMatrixMul / 1000.0f);
    double floatGigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerFloatMatrixMul / 1000.0f);
    fprintf(stderr,
            "%d %.2f %.3f %.2f %.3f\n",
            N,
            floatGigaFlops,
            msecPerFloatMatrixMul,
            doubleGigaFlops,
            msecPerDoubleMatrixMul);
}


/* Generate matrix */
void GPU_fill_rand(double *A, int N){
    curandGenerator_t prng;
    curandCreateGenerator (&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniformDouble(prng, A, N);
}

void ConstantInitDouble(double *data, int size, double val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val+i;
    }
}

void ConstantInitSimple(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/* DEBUG functions */
void print_dcmatrix(const thrust::complex<double>*A, int N, bool rowmajor){
    for(int i = 0; i <N; ++i){
        for(int j = 0; j <N; ++j){
            int ind = rowmajor ? i*N+j: j*N+i;
            std::cerr << A[ind] << " ";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}
void print_fcmatrix(const thrust::complex<float>*A, int N, bool rowmajor){
    for(int i = 0; i <N; ++i){
        for(int j = 0; j <N; ++j){
            int ind = rowmajor ? i*N+j: j*N+i;
            std::cerr << A[ind] << " ";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

void print_dmatrix(const double*A, int N, bool rowmajor){
    for(int i = 0; i <N; ++i){
        for(int j = 0; j <N; ++j){
            int ind = rowmajor ? i*N+j: j*N+i;
            fprintf(stderr,"%.2f ",  A[ind]);
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}
void print_fmatrix(const float*A, int N, bool rowmajor){
    for(int i = 0; i <N; ++i){
        for(int j = 0; j <N; ++j){
            int ind = rowmajor ? i*N+j: j*N+i;
            fprintf(stderr,"%.2f ",  A[ind]);
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}
