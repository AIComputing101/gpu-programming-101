#include <hip/hip_runtime.h>
#include <stdio.h>

// HIP error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(error));
        return -1;
    }
    
    printf("Number of HIP devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        
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
        
        // HIP-specific properties
        printf("  Device Architecture: %s\n", props.gcnArchName);
        printf("  PCI Bus ID: %d\n", props.pciBusID);
        printf("  PCI Device ID: %d\n", props.pciDeviceID);
        printf("  PCI Domain ID: %d\n", props.pciDomainID);
        printf("  Compute Mode: %d\n", props.computeMode);
        printf("  Clock Rate: %.2f GHz\n", props.clockRate / 1e6);
        
        // Check current memory usage
        size_t free_mem, total_mem;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
        printf("  Current Memory Usage: %.2f GB free of %.2f GB total\n", 
               free_mem / (1024.0 * 1024.0 * 1024.0), 
               total_mem / (1024.0 * 1024.0 * 1024.0));
        
        // Platform information
        printf("  Platform: ");
#ifdef __HIP_PLATFORM_AMD__
        printf("AMD ROCm\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
        printf("NVIDIA CUDA\n");
#else
        printf("Unknown\n");
#endif
        
        // Additional ROCm-specific info
        printf("  ISA Name: %s\n", props.gcnArchName);
        printf("  Cooperative Launch: %s\n", props.cooperativeLaunch ? "Yes" : "No");
        printf("  Cooperative Multi-Device Launch: %s\n", props.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        
        printf("\n");
    }
    
    // Show HIP runtime version
    int runtimeVersion;
    if (hipRuntimeGetVersion(&runtimeVersion) == hipSuccess) {
        printf("HIP Runtime Version: %d.%d\n", 
               runtimeVersion / 10000, (runtimeVersion % 10000) / 100);
    }
    
    // Show HIP driver version if available
    int driverVersion;
    if (hipDriverGetVersion(&driverVersion) == hipSuccess) {
        printf("HIP Driver Version: %d.%d\n", 
               driverVersion / 10000, (driverVersion % 10000) / 100);
    }
    
    return 0;
}