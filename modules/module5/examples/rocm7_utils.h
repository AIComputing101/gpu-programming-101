#ifndef ROCM7_UTILS_H
#define ROCM7_UTILS_H

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ROCm 7.0 Enhanced Error Checking Utility
// This header provides improved error handling and debugging capabilities
// specifically designed for ROCm 7.0 features

// Enhanced HIP error checking macro with ROCm 7.0 features
#define HIP_CHECK_ENHANCED(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            const char* errorName = hipGetErrorName(error); \
            const char* errorString = hipGetErrorString(error); \
            fprintf(stderr, "\n=== ROCm 7.0 HIP Error ===\n"); \
            fprintf(stderr, "Error Code: %s (%d)\n", errorName, error); \
            fprintf(stderr, "Error Description: %s\n", errorString); \
            fprintf(stderr, "File: %s\n", __FILE__); \
            fprintf(stderr, "Line: %d\n", __LINE__); \
            fprintf(stderr, "Function: %s\n", __func__); \
            fprintf(stderr, "========================\n"); \
            \
            /* Print device information for context */ \
            int device; \
            if (hipGetDevice(&device) == hipSuccess) { \
                hipDeviceProp_t props; \
                if (hipGetDeviceProperties(&props, device) == hipSuccess) { \
                    fprintf(stderr, "Current Device: %d (%s)\n", device, props.name); \
                    fprintf(stderr, "ROCm Version Support: %d.%d\n", props.major, props.minor); \
                } \
            } \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ROCm 7.0 Memory Management Utilities
inline void hipSafeCleanup(void** ptr) {
    if (ptr && *ptr) {
        hipError_t error = hipFree(*ptr);
        if (error != hipSuccess) {
            fprintf(stderr, "Warning: hipFree failed with error %s\n", hipGetErrorString(error));
        }
        *ptr = nullptr;
    }
}

// ROCm 7.0 Event Management Utilities
inline void hipSafeEventDestroy(hipEvent_t* event) {
    if (event && *event) {
        hipError_t error = hipEventDestroy(*event);
        if (error != hipSuccess) {
            fprintf(stderr, "Warning: hipEventDestroy failed with error %s\n", hipGetErrorString(error));
        }
        *event = nullptr;
    }
}

// ROCm 7.0 Device Information Display
inline void printROCm7DeviceInfo() {
    int deviceCount;
    HIP_CHECK_ENHANCED(hipGetDeviceCount(&deviceCount));
    
    printf("\n=== ROCm 7.0 Device Information ===\n");
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        HIP_CHECK_ENHANCED(hipGetDeviceProperties(&props, i));
        
        printf("Device %d: %s\n", i, props.name);
        printf("  Compute Capability: %d.%d\n", props.major, props.minor);
        printf("  Architecture: %s\n", props.gcnArchName);
        printf("  Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", props.multiProcessorCount);
        printf("  Max Threads per MP: %d\n", props.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", props.warpSize);
        printf("  L2 Cache Size: %d bytes\n", props.l2CacheSize);
        
        // ROCm 7.0 specific features
        printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
        printf("  Memory Clock Rate: %.2f MHz\n", props.memoryClockRate / 1000.0);
        printf("  Concurrent Kernels: %s\n", props.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", props.ECCEnabled ? "Yes" : "No");
        
        size_t free_mem, total_mem;
        HIP_CHECK_ENHANCED(hipSetDevice(i));
        HIP_CHECK_ENHANCED(hipMemGetInfo(&free_mem, &total_mem));
        printf("  Available Memory: %.2f GB / %.2f GB\n", 
               free_mem / (1024.0 * 1024.0 * 1024.0),
               total_mem / (1024.0 * 1024.0 * 1024.0));
        printf("\n");
    }
}

// ROCm 7.0 Performance Timing Utility
class ROCm7Timer {
private:
    hipEvent_t start, stop;
    bool timing_active;

public:
    ROCm7Timer() : timing_active(false) {
        HIP_CHECK_ENHANCED(hipEventCreate(&start));
        HIP_CHECK_ENHANCED(hipEventCreate(&stop));
    }
    
    ~ROCm7Timer() {
        hipSafeEventDestroy(&start);
        hipSafeEventDestroy(&stop);
    }
    
    void startTiming() {
        HIP_CHECK_ENHANCED(hipEventRecord(start, 0));
        timing_active = true;
    }
    
    float stopTiming() {
        if (!timing_active) {
            fprintf(stderr, "Warning: Timer not started\n");
            return 0.0f;
        }
        
        HIP_CHECK_ENHANCED(hipEventRecord(stop, 0));
        HIP_CHECK_ENHANCED(hipEventSynchronize(stop));
        
        float elapsed_ms;
        HIP_CHECK_ENHANCED(hipEventElapsedTime(&elapsed_ms, start, stop));
        timing_active = false;
        
        return elapsed_ms;
    }
};

// Macro for backward compatibility
#define HIP_CHECK HIP_CHECK_ENHANCED

#endif // ROCM7_UTILS_H