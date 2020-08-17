/* Get GPU and CUDA Informations using CUDA with error checking.
Print them in an orderly fashion.
More informations on cudaDeviceProp: https://docs.nvidia.com/cuda/cuda-runtime-api/ */
#include <stdio.h>
#include <stdlib.h>

#define KB 1024
#define MB (KB*KB)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
       printf("Error getting infos: %s", cudaGetErrorString(code));
}

char* getArch(cudaDeviceProp prop)
{
    /* Get arch from prop.major */
    char* archName = (char*)malloc(14*sizeof(char));
    switch(prop.major){
        case 2:
            archName = "Fermi";
            break;
        case 3:
            archName = "Kepler";
            break;
        case 5:
            archName = "Maxwell";
            break;
        case 6:
            archName = "Pascal";
            break;
        case 7:
            archName = "Volta/Turing";
            break;
        case 8:
            archName = "Ampere";
            break;
        default:
            archName = "Unknown Arch.";
            break;
    }
    return archName;
}

int main(void)
{
    int devCount;
    gpuErrchk(cudaGetDeviceCount(&devCount));
    printf("CUDA Version:\t%.2f\n", (float)CUDART_VERSION / 1000);
    for(int i=0; i<devCount; i++)
    {
        /* Get device properties based on index */
        cudaDeviceProp prop;
        gpuErrchk(cudaGetDeviceProperties(&prop, i));

        printf("\n");
        printf("%-40s%d\n", "Device number:", i);
        printf("%-40s%s\n", "Name:", prop.name);
        printf("%-40s%s\n", "Architecture: ", getArch(prop));
        printf("%-40s%.0f Mhz\n", "Clock rate:", (float)prop.clockRate/KB);
        printf("%-40s%d Mhz\n", "Memory clock rate:", prop.memoryClockRate/KB);
        printf("%-40s%.0f Mb\n", "Global mem:", (float)prop.totalGlobalMem/MB);
        printf("%-40s%d\n", "PCI Bus ID:", prop.pciBusID);
    }
}