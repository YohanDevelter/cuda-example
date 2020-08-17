/* Using a dataset consisting of three arrays: A, B and C, the operation
Cx = Ax + Bx is performed on each element.

This code is executed on the GPU using CUDA. The result compute time is
displayed at the end of the computing.*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void device_add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_array(int *data, int size){
    /* Fill an array with the index */
    for(int i=0; i<size; i++)
        data[i] = i;
}

void print_output(int *a, int *b, int *c, int size){
    for(int i=0; i<size; i++)
        printf("\n %d + %d = %d", a[i], b[i], c[i]);
}

int main(int argc, char *argv[]) {
    int sizeOfArray = 512;
    if(argc > 1)
        sizeOfArray = atoi(argv[1]);

    int *a, *b, *c;                     // Host copies of A, B and C
    int *d_a, *d_b, *d_c;               // Device copies of A, B and C
    int memSize = sizeOfArray * sizeof(int);
    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME, &start);

    /* Alloc space for host copies of a, b and c. Setup with input values */
    a = (int*)malloc(memSize); fill_array(a, sizeOfArray);
    b = (int*)malloc(memSize); fill_array(b, sizeOfArray);
    c = (int*)malloc(memSize);

    /* Alloc space on the GPU */
    cudaMalloc((void**)&d_a, memSize);
    cudaMalloc((void**)&d_b, memSize);
    cudaMalloc((void**)&d_c, memSize);

    /* Copy from host to device */
    cudaMemcpy(d_a, a, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice);

    /* Compute and copy results back to host */
    device_add<<<sizeOfArray,1>>>(d_a,d_b,d_c);
    cudaMemcpy(c, d_c, sizeOfArray * sizeof(int), cudaMemcpyDeviceToHost);

    /* Get compute time */
    clock_gettime(CLOCK_REALTIME, &finish);
    long seconds = finish.tv_sec - start.tv_sec;
    long ns = finish.tv_nsec - start.tv_nsec;

    if (start.tv_nsec > finish.tv_nsec) {
        --seconds;
        ns += 1000000000;
    }

    /* Print output */
    print_output(a, b, c, sizeOfArray);
    printf("\n\nTime to compute: %e seconds \n\n",
            (double)seconds + (double)ns/(double)1000000000);

    free(a); free(b); free(c);                      // Free host memory
    cudaFree(d_a), cudaFree(d_b), cudaFree(d_c);    // Free GPU memory
    return 0;
}