#include <cstdlib>
#include "cublas_functions.h"

int main(int argc, char * argv[]){
    int block_size = BSIZE;
    for(int N=2048; N<20000; N += 2048){
        MatrixVectorMultiplyReal(N, block_size);
    }
    //for(int N=256; N<20000; N *= 2){
    //    MatrixMultiplyReal(N, block_size);
    //}
    return 0;
}
