FLAGS=-lineinfo -O2 -arch=compute_50 -code=sm_50

seam_carver:  main.o cuda_kernels.o seam_carver_functions.o
	nvcc -o seam_carver main.o cuda_kernels.o seam_carver_functions.o $(FLAGS)
cuda_kernels.o: cuda_kernels.cu cuda_kernels.h
	nvcc -c cuda_kernels.cu $(FLAGS)
seam_carver_functions.o: seam_carver_functions.c seam_carver.h
	nvcc -c seam_carver_functions.c $(FLAGS)
main.o: main.c
	nvcc -c main.c $(FLAGS)
clean:
	rm -f *.o


