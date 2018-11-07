seam_carver:  main.o cuda_kernels.o
	nvcc -o seam_carver main.o cuda_kernels.o -lineinfo
cuda_kernels.o: cuda_kernels.cu cuda_kernels.h
	nvcc -c cuda_kernels.cu -lineinfo
main.o: main.c
	nvcc -c main.c -lineinfo
clean:
	rm -f *.o


