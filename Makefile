all: mul
cmatmul:
	nvcc -I Common -o $@ main_complex.cu cublas_xgemm_functions.cu matrix_utils.cu -lcublas -lcurand $(DEBUG)
matmul:
	nvcc -I Common -o $@ main.cu cublas_xgemm_functions.cu matrix_utils.cu -lcublas -lcurand $(DEBUG)

mul:
	nvcc -I Common -o $@ main.cu  cublas_xgemm_functions.cu cublas_xgemv_functions.cu matrix_utils.cu -lcublas -lcurand $(DEBUG)

matvecmul:
	nvcc -I Common -o $@ main.cu cublas_xgemv_functions.cu matrix_utils.cu -lcublas -lcurand $(DEBUG)

clean:
	rm -f cmatmul matmul simple matvecmul

.PHONY: clean
