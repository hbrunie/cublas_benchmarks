#include <cstdlib>
#include "gemm_cublas_complex_functions.h"

int main(int argc, char * argv[]){
    int block_size = BSIZE;
    for(int N=BSIZE; N<25*BSIZE; N+=2*BSIZE){
        N = N - (N%block_size);
        MatrixMultiplyComplex(N,BSIZE);
    }
    return 0;
}
