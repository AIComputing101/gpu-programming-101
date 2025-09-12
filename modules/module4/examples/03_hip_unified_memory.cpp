#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>

#define ARRAY_SIZE (16 * 1024 * 1024)  // 16M elements
#define BLOCK_SIZE 256

// Simple computation kernel
__global__ void computeKernel(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < iterations; i++) {
            value = sinf(value * 1.1f) + cosf(value * 0.9f);
        }
        data[idx] = value;
    }
}

// Vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Reduction kernel
__global__ void reductionKernel(float *input, float *output, int n) {
    HIP_DYNAMIC_SHARED(float, shared)
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void printDeviceInfo() {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    printf("=== HIP Unified Memory Device Information ===\n");
    printf("Number of devices: %d\n", deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        HIP_CHECK(hipSetDevice(device));
        
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));
        
        printf("\nDevice %d: %s\n", device, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Unified Memory Support: ");
        
        // Check unified memory support (HIP version dependent)
        int unifiedAddressing;
        HIP_CHECK(hipDeviceGetAttribute(&unifiedAddressing, hipDeviceAttributeUnifiedAddressing, device));
        printf("%s\n", unifiedAddressing ? "Yes" : "No");
        
        // Check managed memory support
        int managedMemory;
        hipError_t result = hipDeviceGetAttribute(&managedMemory, hipDeviceAttributeManagedMemory, device);
        if (result == hipSuccess) {
            printf("  Managed Memory Support: %s\n", managedMemory ? "Yes" : "No");
        } else {
            printf("  Managed Memory Support: Unknown (older HIP version)\n");
        }
        
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Memory Clock Rate: %.1f GHz\n", prop.memoryClockRate / 1e6);
    }
    printf("\n");
}

// Traditional explicit memory management
double runTraditionalMemory(float *h_input, float *h_output, int size) {
    printf("Running traditional explicit memory management...\n");
    
    float *d_input, *d_output;
    size_t bytes = size * sizeof(float);
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);
    hipLaunchKernelGGL(computeKernel, grid, block, 0, 0, d_input, size, 100);
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_output, d_input, bytes, hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    
    double timeMs = duration.count() / 1000.0;
    printf("Traditional memory time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Basic unified memory usage
double runUnifiedMemoryBasic(float *data, int size) {
    printf("Running basic unified memory...\n");
    
    float *unified_data;
    size_t bytes = size * sizeof(float);
    
    // Allocate unified memory
    HIP_CHECK(hipMallocManaged(&unified_data, bytes));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        unified_data[i] = data[i];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel - no explicit memory transfers needed
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);
    hipLaunchKernelGGL(computeKernel, grid, block, 0, 0, unified_data, size, 100);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Access result data directly (may cause page faults)
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {  // Sample first 1000 elements
        sum += unified_data[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Sample sum: %.6f\n", sum);
    
    // Copy result back to original array
    for (int i = 0; i < size; i++) {
        data[i] = unified_data[i];
    }
    
    HIP_CHECK(hipFree(unified_data));
    
    double timeMs = duration.count() / 1000.0;
    printf("Basic unified memory time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Optimized unified memory with prefetching and hints
double runUnifiedMemoryOptimized(float *data, int size) {
    printf("Running optimized unified memory with prefetching...\n");
    
    float *unified_data;
    size_t bytes = size * sizeof(float);
    
    int device;
    HIP_CHECK(hipGetDevice(&device));
    
    // Allocate unified memory
    HIP_CHECK(hipMallocManaged(&unified_data, bytes));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        unified_data[i] = data[i];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Provide memory hints (HIP may not support all CUDA hints)
    hipError_t result;
    
    // Try to set memory advice (may not be supported on all platforms)
    result = hipMemAdvise(unified_data, bytes, hipMemAdviseSetReadMostly, device);
    if (result == hipSuccess) {
        printf("  Set read-mostly advice\n");
    }
    
    result = hipMemAdvise(unified_data, bytes, hipMemAdviseSetPreferredLocation, device);
    if (result == hipSuccess) {
        printf("  Set preferred location to GPU\n");
    }
    
    // Prefetch data to GPU
    result = hipMemPrefetchAsync(unified_data, bytes, device, 0);
    if (result == hipSuccess) {
        printf("  Prefetched data to GPU\n");
    } else {
        printf("  Prefetching not supported on this platform\n");
    }
    
    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);
    hipLaunchKernelGGL(computeKernel, grid, block, 0, 0, unified_data, size, 100);
    
    // Prefetch result back to CPU before accessing
    result = hipMemPrefetchAsync(unified_data, bytes, hipCpuDeviceId, 0);
    if (result == hipSuccess) {
        printf("  Prefetched data back to CPU\n");
    }
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Access result data
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += unified_data[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Sample sum: %.6f\n", sum);
    
    // Copy result back
    for (int i = 0; i < size; i++) {
        data[i] = unified_data[i];
    }
    
    HIP_CHECK(hipFree(unified_data));
    
    double timeMs = duration.count() / 1000.0;
    printf("Optimized unified memory time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Multi-GPU unified memory example
double runMultiGPUUnified(float *data, int size) {
    printf("Running multi-GPU unified memory example...\n");
    
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount < 2) {
        printf("Multi-GPU test requires at least 2 GPUs. Found: %d\n", deviceCount);
        printf("Skipping multi-GPU unified memory test.\n\n");
        return 0.0;
    }
    
    float *unified_a, *unified_b, *unified_c;
    size_t bytes = size * sizeof(float);
    
    // Allocate unified memory
    HIP_CHECK(hipMallocManaged(&unified_a, bytes));
    HIP_CHECK(hipMallocManaged(&unified_b, bytes));
    HIP_CHECK(hipMallocManaged(&unified_c, bytes));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        unified_a[i] = data[i];
        unified_b[i] = sinf(i * 0.001f);
        unified_c[i] = 0.0f;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use different GPUs for different operations
    int chunkSize = size / deviceCount;
    std::vector<hipStream_t> streams(deviceCount);
    
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        HIP_CHECK(hipSetDevice(gpu));
        HIP_CHECK(hipStreamCreate(&streams[gpu]));
        
        int offset = gpu * chunkSize;
        int currentChunkSize = (gpu == deviceCount - 1) ? 
                              size - offset : chunkSize;
        
        // Prefetch data to current GPU
        hipMemPrefetchAsync(unified_a + offset, currentChunkSize * sizeof(float), gpu, streams[gpu]);
        hipMemPrefetchAsync(unified_b + offset, currentChunkSize * sizeof(float), gpu, streams[gpu]);
        hipMemPrefetchAsync(unified_c + offset, currentChunkSize * sizeof(float), gpu, streams[gpu]);
        
        // Launch vector addition on current GPU
        dim3 block(BLOCK_SIZE);
        dim3 grid((currentChunkSize + block.x - 1) / block.x);
        hipLaunchKernelGGL(vectorAdd, grid, block, 0, streams[gpu],
                          unified_a + offset, unified_b + offset, unified_c + offset, currentChunkSize);
    }
    
    // Synchronize all GPUs
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        HIP_CHECK(hipSetDevice(gpu));
        HIP_CHECK(hipStreamSynchronize(streams[gpu]));
        HIP_CHECK(hipStreamDestroy(streams[gpu]));
    }
    
    // Verify result on CPU
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += unified_c[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Used %d GPUs for computation\n", deviceCount);
    printf("Sample sum: %.6f\n", sum);
    
    // Copy result back
    for (int i = 0; i < size; i++) {
        data[i] = unified_c[i];
    }
    
    HIP_CHECK(hipFree(unified_a));
    HIP_CHECK(hipFree(unified_b));
    HIP_CHECK(hipFree(unified_c));
    
    double timeMs = duration.count() / 1000.0;
    printf("Multi-GPU unified memory time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Iterative CPU-GPU algorithm using unified memory
double runIterativeCPUGPU(float *data, int size, int iterations) {
    printf("Running iterative CPU-GPU algorithm (%d iterations)...\n", iterations);
    
    float *unified_data;
    size_t bytes = size * sizeof(float);
    
    HIP_CHECK(hipMallocManaged(&unified_data, bytes));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        unified_data[i] = data[i];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);
    
    for (int iter = 0; iter < iterations; iter++) {
        // GPU computation phase
        hipLaunchKernelGGL(computeKernel, grid, block, 0, 0, unified_data, size, 10);
        HIP_CHECK(hipDeviceSynchronize());
        
        // CPU processing phase - scale values
        for (int i = 0; i < size; i += 1000) {  // Process every 1000th element
            unified_data[i] *= 0.99f;  // Prevent overflow
        }
        
        // Optional: Print progress
        if ((iter + 1) % (iterations / 4) == 0) {
            printf("  Completed iteration %d/%d\n", iter + 1, iterations);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify final result
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += unified_data[i];
    }
    printf("Final sample sum: %.6f\n", sum);
    
    // Copy result back
    for (int i = 0; i < size; i++) {
        data[i] = unified_data[i];
    }
    
    HIP_CHECK(hipFree(unified_data));
    
    double timeMs = duration.count() / 1000.0;
    printf("Iterative CPU-GPU time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

bool verifyResults(float *reference, float *result, int size, float tolerance = 1e-3f) {
    int errors = 0;
    for (int i = 0; i < size && errors < 10; i++) {
        if (fabsf(reference[i] - result[i]) > tolerance) {
            printf("Verification error at index %d: ref=%.6f, result=%.6f\n", 
                   i, reference[i], result[i]);
            errors++;
        }
    }
    return errors == 0;
}

void demonstratePageFaultBehavior(int size) {
    printf("=== Demonstrating Page Fault Behavior ===\n");
    
    float *managed_data;
    size_t bytes = size * sizeof(float);
    
    HIP_CHECK(hipMallocManaged(&managed_data, bytes));
    
    printf("Allocated %d MB of managed memory\n", (int)(bytes / (1024*1024)));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // First access - will cause page faults
    printf("First CPU access (will cause page faults)...\n");
    for (int i = 0; i < size; i++) {
        managed_data[i] = sinf(i * 0.001f);
    }
    
    auto cpu_init_end = std::chrono::high_resolution_clock::now();
    
    // GPU access - may cause page faults if data is on CPU
    printf("GPU access (potential page faults)...\n");
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);
    hipLaunchKernelGGL(computeKernel, grid, block, 0, 0, managed_data, size, 50);
    HIP_CHECK(hipDeviceSynchronize());
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    
    // Second CPU access
    printf("Second CPU access (potential page faults back to CPU)...\n");
    float sum = 0.0f;
    for (int i = 0; i < size; i += 100) {
        sum += managed_data[i];
    }
    
    auto cpu_access_end = std::chrono::high_resolution_clock::now();
    
    printf("CPU initialization: %.2f ms\n", 
           std::chrono::duration_cast<std::chrono::microseconds>(cpu_init_end - start).count() / 1000.0);
    printf("GPU computation: %.2f ms\n", 
           std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - cpu_init_end).count() / 1000.0);
    printf("CPU access: %.2f ms\n", 
           std::chrono::duration_cast<std::chrono::microseconds>(cpu_access_end - gpu_end).count() / 1000.0);
    printf("Sample sum: %.6f\n", sum);
    
    HIP_CHECK(hipFree(managed_data));
    printf("\n");
}

int main() {
    printf("=== HIP Unified Memory Programming Demo ===\n\n");
    
    // Print device information
    printDeviceInfo();
    
    // Initialize test data
    printf("Initializing test data (%.1f MB)...\n", 
           ARRAY_SIZE * sizeof(float) / (1024.0f * 1024.0f));
    
    float *h_input = new float[ARRAY_SIZE];
    float *h_traditional = new float[ARRAY_SIZE];
    float *h_unified_basic = new float[ARRAY_SIZE];
    float *h_unified_optimized = new float[ARRAY_SIZE];
    float *h_multi_gpu = new float[ARRAY_SIZE];
    float *h_iterative = new float[ARRAY_SIZE];
    
    // Initialize input data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        float value = sinf(i * 0.001f) * cosf(i * 0.0005f);
        h_input[i] = value;
        h_traditional[i] = value;
        h_unified_basic[i] = value;
        h_unified_optimized[i] = value;
        h_multi_gpu[i] = value;
        h_iterative[i] = value;
    }
    
    printf("\n=== Performance Comparison ===\n");
    
    // Traditional memory management
    double traditionalTime = runTraditionalMemory(h_input, h_traditional, ARRAY_SIZE);
    
    // Basic unified memory
    double basicTime = runUnifiedMemoryBasic(h_unified_basic, ARRAY_SIZE);
    
    // Optimized unified memory
    double optimizedTime = runUnifiedMemoryOptimized(h_unified_optimized, ARRAY_SIZE);
    
    // Multi-GPU unified memory
    double multiGPUTime = runMultiGPUUnified(h_multi_gpu, ARRAY_SIZE);
    
    // Iterative CPU-GPU algorithm
    double iterativeTime = runIterativeCPUGPU(h_iterative, ARRAY_SIZE, 10);
    
    // Demonstrate page fault behavior
    demonstratePageFaultBehavior(1024 * 1024);  // 1M elements
    
    // Verification
    printf("=== Verification ===\n");
    bool basicValid = verifyResults(h_traditional, h_unified_basic, ARRAY_SIZE);
    bool optimizedValid = verifyResults(h_traditional, h_unified_optimized, ARRAY_SIZE);
    
    printf("Traditional vs Basic unified: %s\n", basicValid ? "VALID" : "INVALID");
    printf("Traditional vs Optimized unified: %s\n", optimizedValid ? "VALID" : "INVALID");
    printf("\n");
    
    // Performance analysis
    printf("=== Performance Analysis ===\n");
    printf("Traditional memory: %.2f ms\n", traditionalTime);
    printf("Basic unified memory: %.2f ms (%.1fx %s)\n", 
           basicTime, fabsf(traditionalTime / basicTime), 
           basicTime < traditionalTime ? "faster" : "slower");
    printf("Optimized unified memory: %.2f ms (%.1fx %s)\n", 
           optimizedTime, fabsf(traditionalTime / optimizedTime),
           optimizedTime < traditionalTime ? "faster" : "slower");
    
    if (multiGPUTime > 0) {
        printf("Multi-GPU unified memory: %.2f ms (%.1fx %s)\n", 
               multiGPUTime, fabsf(traditionalTime / multiGPUTime),
               multiGPUTime < traditionalTime ? "faster" : "slower");
    }
    
    printf("Iterative CPU-GPU: %.2f ms (10 iterations)\n", iterativeTime);
    printf("\n");
    
    printf("=== Key Insights ===\n");
    printf("- HIP unified memory simplifies GPU programming by eliminating explicit transfers\n");
    printf("- Page faults can impact performance on first access\n");
    printf("- Memory hints and prefetching can optimize performance\n");
    printf("- Unified memory works across multiple GPUs seamlessly\n");
    printf("- Ideal for iterative CPU-GPU algorithms\n");
    
    if (optimizedTime < basicTime) {
        printf("- Prefetching and hints provided %.1fx speedup\n", basicTime / optimizedTime);
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_traditional;
    delete[] h_unified_basic;
    delete[] h_unified_optimized;
    delete[] h_multi_gpu;
    delete[] h_iterative;
    
    printf("\n=== HIP Unified Memory Demo Completed Successfully ===\n");
    return 0;
}