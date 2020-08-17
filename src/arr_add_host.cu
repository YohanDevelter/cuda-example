/* Using a dataset consisting of three arrays: A, B and C, the operation
Cx = Ax + Bx is performed on each element.

This code is executed on the host using the CPU. The result compute time is
displayed at the end of the computing.*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void host_add(int *a, int *b, int *c, int size){
    /* Add two numbers and put them in an array of C (Host computation)
        a, b and c must be arrays */
    for(int i=0; i<size; i++)
        c[i] = a[i] + b[i];
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

    int *a, *b, *c;
    int size = sizeOfArray * sizeof(int);
    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME, &start);

    /* Alloc space for host copies of a, b and c. Setup with input values */
    a = (int*)malloc(size); fill_array(a, sizeOfArray);
    b = (int*)malloc(size); fill_array(b, sizeOfArray);
    c = (int*)malloc(size);
    host_add(a, b, c, sizeOfArray);

    /* Get compute time */
    clock_gettime(CLOCK_REALTIME, &finish);
    long ns = finish.tv_nsec - start.tv_nsec;

    /* Print output */
    print_output(a, b, c, sizeOfArray);
    printf("\n\nTime to compute: %ld ns \n\n", ns);

    free(a);
    free(b);
    free(c);
    return 0;
}