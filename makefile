seam_carver:  main.o cuda_kernels.o
	nvcc -o seam_carver main.o cuda_kernels.o
cuda_kernels.o: cuda_kernels.cu cuda_kernels.h
	nvcc -c cuda_kernels.cu
main.o: main.c
	nvcc -c main.c
clean:
	rm -f *.o


