#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Cross-platform kernel that works on both AMD and NVIDIA
__global__ void crossPlatformKernel(float *input, float *output, int n) {
    // Using HIP built-in variables (works on both platforms)
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    if (tid < n) {
        // Platform-agnostic computation
        float value = input[tid];
        
        // Some mathematical operations
        value = sinf(value) * cosf(value);
        value = sqrtf(value * value + 1.0f);
        
        output[tid] = value;
    }
}

// Platform-specific optimized versions
#ifdef __HIP_PLATFORM_AMD__
__global__ void amdOptimizedKernel(float *input, float *output, int n) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    if (tid < n) {
        // AMD-specific optimizations
        float value = input[tid];
        
        // Use AMD's native math functions when available
        value = __sinf(value) * __cosf(value);  // Fast intrinsic versions
        value = __fsqrt_rn(value * value + 1.0f);
        
        output[tid] = value;
    }
}
#elif defined(__HIP_PLATFORM_NVIDIA__)
__global__ void nvidiaOptimizedKernel(float *input, float *output, int n) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    if (tid < n) {
        // NVIDIA-specific optimizations
        float value = input[tid];
        
        // Use NVIDIA's fast math functions
        value = __sinf(value) * __cosf(value);
        value = __fsqrt_rn(value * value + 1.0f);
        
        output[tid] = value;
    }
}
#endif

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void printPlatformInfo() {
    printf("=== Platform Detection ===\n");
    
#ifdef __HIP_PLATFORM_AMD__
    printf("Detected Platform: AMD ROCm\n");
    printf("HIP Backend: AMD\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("Detected Platform: NVIDIA CUDA\n");
    printf("HIP Backend: NVIDIA\n");
#else
    printf("Unknown Platform\n");
#endif

    // Get device information
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("Number of devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, 0));
        
        printf("Device 0: %s\n", props.name);
        printf("Compute Capability: %d.%d\n", props.major, props.minor);
        printf("Multiprocessors: %d\n", props.multiProcessorCount);
        printf("Warp Size: %d\n", props.warpSize);
        
#ifdef __HIP_PLATFORM_AMD__
        printf("GCN Architecture: %s\n", props.gcnArchName);
        printf("Compute Units: %d\n", props.multiProcessorCount);
#endif
    }
    printf("\n");
}

float benchmarkKernel(void(*kernel)(float*, float*, int), 
                     const char* name, 
                     float *d_input, float *d_output, int n) {
    
    printf("Benchmarking %s kernel...\n", name);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warm up
    for (int i = 0; i < 5; i++) {
        if (kernel == (void(*)(float*, float*, int))crossPlatformKernel) {
            crossPlatformKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#ifdef __HIP_PLATFORM_AMD__
        else if (kernel == (void(*)(float*, float*, int))amdOptimizedKernel) {
            amdOptimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#elif defined(__HIP_PLATFORM_NVIDIA__)
        else if (kernel == (void(*)(float*, float*, int))nvidiaOptimizedKernel) {
            nvidiaOptimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#endif
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    const int iterations = 100;
    HIP_CHECK(hipEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        if (kernel == (void(*)(float*, float*, int))crossPlatformKernel) {
            crossPlatformKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#ifdef __HIP_PLATFORM_AMD__
        else if (kernel == (void(*)(float*, float*, int))amdOptimizedKernel) {
            amdOptimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#elif defined(__HIP_PLATFORM_NVIDIA__)
        else if (kernel == (void(*)(float*, float*, int))nvidiaOptimizedKernel) {
            nvidiaOptimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        }
#endif
    }
    
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float totalTime;
    HIP_CHECK(hipEventElapsedTime(&totalTime, start, stop));
    float avgTime = totalTime / iterations;
    
    printf("  Average time: %.3f ms\n", avgTime);
    printf("  Throughput: %.2f GFLOPS\n", (n * 6.0f) / (avgTime * 1e6f)); // ~6 operations per element
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return avgTime;
}

int main() {
    printf("HIP Cross-Platform Optimization Comparison\n");
    printf("==========================================\n\n");
    
    printPlatformInfo();
    
    const int N = 1024 * 1024; // 1M elements
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / 1000.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    
    printf("=== Performance Comparison ===\n");
    
    // Test cross-platform kernel
    float crossPlatformTime = benchmarkKernel(
        (void(*)(float*, float*, int))crossPlatformKernel,
        "Cross-platform", d_input, d_output, N);
    
    // Test platform-specific optimized version
    float optimizedTime = 0.0f;
#ifdef __HIP_PLATFORM_AMD__
    optimizedTime = benchmarkKernel(
        (void(*)(float*, float*, int))amdOptimizedKernel,
        "AMD-optimized", d_input, d_output, N);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    optimizedTime = benchmarkKernel(
        (void(*)(float*, float*, int))nvidiaOptimizedKernel,
        "NVIDIA-optimized", d_input, d_output, N);
#endif
    
    if (optimizedTime > 0.0f) {
        float speedup = crossPlatformTime / optimizedTime;
        printf("\nOptimization speedup: %.2fx\n", speedup);
    }
    
    // Verify correctness
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = sinf(h_input[i]) * cosf(h_input[i]);
        expected = sqrtf(expected * expected + 1.0f);
        if (fabsf(h_output[i] - expected) > 1e-5f) {
            correct = false;
            printf("Verification failed at element %d: expected %f, got %f\n", 
                   i, expected, h_output[i]);
            break;
        }
    }
    
    if (correct) {
        printf("✓ Results verified correctly\n");
    }
    
    // Platform-specific feature detection
    printf("\n=== Platform-Specific Features ===\n");
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm Features:\n");
    printf("  - Cooperative launch: %s\n", props.cooperativeLaunch ? "Yes" : "No");
    printf("  - Cooperative multi-device: %s\n", props.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  - Architecture: %s\n", props.gcnArchName);
    
    // Check for specific AMD optimizations
    printf("  - Fast math functions: Available\n");
    printf("  - Wavefront size: %d\n", props.warpSize);
    
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA Features:\n");
    printf("  - Unified addressing: %s\n", props.unifiedAddressing ? "Yes" : "No");
    printf("  - ECC support: %s\n", props.ECCEnabled ? "Yes" : "No");
    printf("  - Concurrent kernels: %s\n", props.concurrentKernels ? "Yes" : "No");
    
    // Check for NVIDIA-specific features
    printf("  - Warp size: %d\n", props.warpSize);
    printf("  - Async engine count: %d\n", props.asyncEngineCount);
#endif
    
    // Memory bandwidth test
    printf("\n=== Memory Bandwidth Test ===\n");
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Test host-to-device bandwidth
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float h2d_time;
    HIP_CHECK(hipEventElapsedTime(&h2d_time, start, stop));
    float h2d_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (h2d_time / 1000.0f);
    
    // Test device-to-host bandwidth
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float d2h_time;
    HIP_CHECK(hipEventElapsedTime(&d2h_time, start, stop));
    float d2h_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2h_time / 1000.0f);
    
    printf("Host-to-Device Bandwidth: %.2f GB/s\n", h2d_bandwidth);
    printf("Device-to-Host Bandwidth: %.2f GB/s\n", d2h_bandwidth);
    
    // Theoretical peak bandwidth
    float theoretical = 2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6f;
    printf("Theoretical Peak: %.2f GB/s\n", theoretical);
    printf("H2D Efficiency: %.1f%%\n", (h2d_bandwidth / theoretical) * 100.0f);
    printf("D2H Efficiency: %.1f%%\n", (d2h_bandwidth / theoretical) * 100.0f);
    
    // Cleanup
    free(h_input);
    free(h_output);
    hipFree(d_input);
    hipFree(d_output);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    printf("\n=== Summary ===\n");
    printf("This example demonstrates:\n");
    printf("  1. Writing portable HIP code that works on AMD and NVIDIA\n");
    printf("  2. Platform-specific optimizations\n");
    printf("  3. Performance comparison between generic and optimized code\n");
    printf("  4. Platform feature detection\n");
    printf("  5. Memory bandwidth analysis\n");
    
    return 0;
}