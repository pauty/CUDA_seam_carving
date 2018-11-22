seam_carver:  main.o cuda_kernels.o seam_carver_functions.o
	nvcc -o seam_carver main.o cuda_kernels.o seam_carver_functions.o -lineinfo -O3 -arch=compute_50 -code=sm_50
cuda_kernels.o: cuda_kernels.cu cuda_kernels.h
	nvcc -c cuda_kernels.cu -lineinfo -O3 -arch=compute_50 -code=sm_50
seam_carver_functions.o: seam_carver_functions.c seam_carver.h
	nvcc -c seam_carver_functions.c -lineinfo -O3 -arch=compute_50 -code=sm_50
main.o: main.c
	nvcc -c main.c -lineinfo -O3 -arch=compute_50 -code=sm_50
clean:
	rm -f *.o


