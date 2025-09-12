#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        
        printf("Device %d: %s\n", i, props.name);
        printf("  Compute Capability: %d.%d\n", props.major, props.minor);
        printf("  Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu bytes\n", props.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", props.regsPerBlock);
        printf("  Warp Size: %d\n", props.warpSize);
        printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
        printf("  Max Threads Dimensions: (%d, %d, %d)\n", 
               props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        printf("  Memory Clock Rate: %.2f GHz\n", props.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
        printf("  Multiprocessor Count: %d\n", props.multiProcessorCount);
        printf("  L2 Cache Size: %d bytes\n", props.l2CacheSize);
        printf("  Max Threads per Multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
        printf("  Concurrent Kernels: %s\n", props.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", props.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", props.unifiedAddressing ? "Yes" : "No");
        
        // Check current memory usage
        size_t free_mem, total_mem;
        cudaSetDevice(i);
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("  Current Memory Usage: %.2f GB free of %.2f GB total\n", 
               free_mem / (1024.0 * 1024.0 * 1024.0), 
               total_mem / (1024.0 * 1024.0 * 1024.0));
        
        printf("\n");
    }
    
    return 0;
}