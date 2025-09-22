#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define ARRAY_SIZE (32 * 1024 * 1024)  // 32M elements
#define BLOCK_SIZE 256

// Simple vector addition kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Memory-intensive kernel to test page fault behavior
__global__ void memoryIntensiveKernel(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Access memory with specified stride to control page fault patterns
        int access_idx = (idx * stride) % n;
        data[access_idx] *= 2.0f;
    }
}

// CPU-intensive computation
__global__ void computeIntensiveKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < 1000; i++) {
            value = sinf(value) + cosf(value * 0.5f);
        }
        data[idx] = value;
    }
}

// Iterative CPU-GPU computation pattern
void hybridComputation(float *data, int n, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        // CPU work
        #pragma omp parallel for
        for (int i = 0; i < n; i += 1000) {
            data[i] = data[i] * 0.99f + iter * 0.01f;
        }
        
        // GPU work
        dim3 block(BLOCK_SIZE);
        dim3 grid((n + block.x - 1) / block.x);
        computeIntensiveKernel<<<grid, block>>>(data, n);
        cudaDeviceSynchronize();
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void checkUnifiedMemorySupport() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Unified Memory Support Analysis:\n");
    printf("===============================\n");
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Managed Memory: %s\n", prop.managedMemory ? "Supported" : "Not Supported");
        printf("  Concurrent Managed Access: %s\n", 
               prop.concurrentManagedAccess ? "Supported" : "Not Supported");
        printf("  Page Lockable Host Memory: %s\n", 
               prop.pageableMemoryAccess ? "Supported" : "Not Supported");
        
        // Check memory pool support (CUDA 11.2+)
        int memPoolSupport;
        CUDA_CHECK(cudaDeviceGetAttribute(&memPoolSupport, 
                                         cudaDevAttrMemoryPoolsSupported, dev));
        printf("  Memory Pools: %s\n", memPoolSupport ? "Supported" : "Not Supported");
        
        printf("\n");
    }
}

// Traditional explicit memory management baseline
double explicitMemoryManagement(int n) {
    printf("Running explicit memory management baseline...\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_a[i] = sinf(i * 0.01f);
        h_b[i] = cosf(i * 0.01f);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return time;
}

// Basic unified memory usage
double basicUnifiedMemory(int n) {
    printf("Running basic unified memory implementation...\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate unified memory
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    
    // Initialize data on CPU
    for (int i = 0; i < n; i++) {
        a[i] = sinf(i * 0.01f);
        b[i] = cosf(i * 0.01f);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel (no explicit memory transfers needed)
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Access result on CPU (automatic migration)
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += c[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    cudaFree(a); cudaFree(b); cudaFree(c);
    
    return time;
}

// Optimized unified memory with prefetching and hints
double optimizedUnifiedMemory(int n) {
    printf("Running optimized unified memory with prefetching...\n");
    
    size_t bytes = n * sizeof(float);
    
    // Allocate unified memory
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    
    // Initialize data on CPU
    for (int i = 0; i < n; i++) {
        a[i] = sinf(i * 0.01f);
        b[i] = cosf(i * 0.01f);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Provide memory hints (CUDA 13: use cudaMemLocation)
    int deviceId = 0;
    cudaMemLocation locDevice{};
    locDevice.type = cudaMemLocationTypeDevice;
    locDevice.id = deviceId;
    cudaMemLocation locHost{};
    locHost.type = cudaMemLocationTypeHost;
    locHost.id = 0;

    CUDA_CHECK(cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, locDevice));
    CUDA_CHECK(cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, locDevice));
    
    // Prefetch data to GPU (location-aware + explicit stream)
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, locDevice, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, locDevice, 0));
    
    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    
    // Prefetch result back to CPU (location-aware + explicit stream)
    CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, locHost, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Access result on CPU
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += c[i];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    cudaFree(a); cudaFree(b); cudaFree(c);
    
    return time;
}

// Demonstrate page fault behavior
void demonstratePageFaultBehavior(int n) {
    printf("\nDemonstrating Page Fault Behavior:\n");
    printf("==================================\n");
    
    size_t bytes = n * sizeof(float);
    float *data;
    
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    // Initialize data on CPU
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
    
    printf("Testing different memory access patterns:\n\n");
    
    // Test 1: Sequential access (good for page faults)
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);
    memoryIntensiveKernel<<<grid, block>>>(data, n, 1); // stride = 1
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double sequential_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Sequential access (stride=1): %.2f ms\n", sequential_time);
    
    // Test 2: Strided access (causes more page faults)
    start = std::chrono::high_resolution_clock::now();
    
    memoryIntensiveKernel<<<grid, block>>>(data, n, 1024); // stride = 1024
    CUDA_CHECK(cudaDeviceSynchronize());
    
    end = std::chrono::high_resolution_clock::now();
    double strided_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Strided access (stride=1024): %.2f ms (%.2fx slower)\n", 
           strided_time, strided_time / sequential_time);
    
    // Test 3: Random access (worst case for page faults)
    start = std::chrono::high_resolution_clock::now();
    
    memoryIntensiveKernel<<<grid, block>>>(data, n, 12345); // stride = 12345 (pseudo-random)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    end = std::chrono::high_resolution_clock::now();
    double random_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Random access (stride=12345): %.2f ms (%.2fx slower)\n", 
           random_time, random_time / sequential_time);
    
    cudaFree(data);
    
    printf("\nKey Insight: Page fault overhead depends heavily on access patterns.\n");
    printf("Sequential access minimizes page faults, while random access maximizes them.\n");
}

// Hybrid CPU-GPU computation with unified memory
double hybridCPUGPUComputation(int n) {
    printf("\nRunning hybrid CPU-GPU computation...\n");
    
    size_t bytes = n * sizeof(float);
    float *data;
    
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        data[i] = sinf(i * 0.001f);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform hybrid computation
    hybridComputation(data, n, 5); // 5 iterations
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Hybrid computation completed in %.2f ms\n", time);
    
    cudaFree(data);
    
    return time;
}

// Multi-GPU unified memory demonstration
void multiGPUUnifiedMemory(int n) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount < 2) {
        printf("Multi-GPU unified memory requires at least 2 GPUs.\n");
        return;
    }
    
    printf("\nMulti-GPU Unified Memory Demonstration:\n");
    printf("======================================\n");
    
    size_t bytes = n * sizeof(float);
    float *data;
    
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
    
    // Enable peer access between GPUs
    for (int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                if (canAccessPeer) {
                    cudaDeviceEnablePeerAccess(j, 0); // Ignore errors
                }
            }
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process data on multiple GPUs simultaneously
    int chunkSize = n / deviceCount;
    
    #pragma omp parallel for
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        int offset = gpu * chunkSize;
        int currentChunkSize = (gpu == deviceCount - 1) ? n - offset : chunkSize;
        
    // Prefetch chunk to current GPU (location-aware + explicit stream)
    cudaMemLocation locGpu{};
    locGpu.type = cudaMemLocationTypeDevice;
    locGpu.id = gpu;
    CUDA_CHECK(cudaMemPrefetchAsync(data + offset,
                       currentChunkSize * sizeof(float),
                       locGpu, 0));
        
        // Process on this GPU
        dim3 block(BLOCK_SIZE);
        dim3 grid((currentChunkSize + block.x - 1) / block.x);
        computeIntensiveKernel<<<grid, block>>>(data + offset, currentChunkSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Multi-GPU processing completed in %.2f ms\n", time);
    printf("Used %d GPUs with unified memory\n", deviceCount);
    
    cudaFree(data);
}

// Memory usage statistics
void analyzeMemoryUsage(int n) {
    printf("\nMemory Usage Analysis:\n");
    printf("=====================\n");
    
    size_t bytes = n * sizeof(float);
    
    // Get initial memory info
    size_t free_before, total_before;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));
    
    printf("Before allocation:\n");
    printf("  Free: %.2f MB, Total: %.2f MB\n", 
           free_before / (1024.0 * 1024.0), total_before / (1024.0 * 1024.0));
    
    // Allocate unified memory
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    size_t free_after, total_after;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
    
    printf("After unified memory allocation (%.2f MB):\n", 
           bytes / (1024.0 * 1024.0));
    printf("  Free: %.2f MB, Total: %.2f MB\n", 
           free_after / (1024.0 * 1024.0), total_after / (1024.0 * 1024.0));
    printf("  Actually used: %.2f MB\n", 
           (free_before - free_after) / (1024.0 * 1024.0));
    
    // Initialize data to trigger physical allocation
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
    
    size_t free_initialized, total_initialized;
    CUDA_CHECK(cudaMemGetInfo(&free_initialized, &total_initialized));
    
    printf("After initialization:\n");
    printf("  Free: %.2f MB\n", free_initialized / (1024.0 * 1024.0));
    printf("  Additional memory used: %.2f MB\n", 
           (free_after - free_initialized) / (1024.0 * 1024.0));
    
    cudaFree(data);
    
    size_t free_final, total_final;
    CUDA_CHECK(cudaMemGetInfo(&free_final, &total_final));
    
    printf("After deallocation:\n");
    printf("  Free: %.2f MB (recovered: %.2f MB)\n", 
           free_final / (1024.0 * 1024.0),
           (free_final - free_before) / (1024.0 * 1024.0));
}

int main() {
    printf("CUDA Unified Memory Demonstration\n");
    printf("=================================\n\n");
    
    checkUnifiedMemorySupport();
    
    const int n = ARRAY_SIZE;
    const size_t bytes = n * sizeof(float);
    
    printf("Array size: %d elements (%.2f MB)\n\n", n, bytes / (1024.0f * 1024.0f));
    
    // Performance comparison
    printf("=== Performance Comparison ===\n");
    
    double explicit_time = explicitMemoryManagement(n);
    double basic_unified_time = basicUnifiedMemory(n);
    double optimized_unified_time = optimizedUnifiedMemory(n);
    
    printf("\nPerformance Results:\n");
    printf("Explicit Memory Management: %.2f ms\n", explicit_time);
    printf("Basic Unified Memory:       %.2f ms (%.2fx vs explicit)\n", 
           basic_unified_time, basic_unified_time / explicit_time);
    printf("Optimized Unified Memory:   %.2f ms (%.2fx vs explicit)\n", 
           optimized_unified_time, optimized_unified_time / explicit_time);
    
    // Demonstrate page fault behavior
    demonstratePageFaultBehavior(n / 8); // Use smaller size for page fault demo
    
    // Hybrid CPU-GPU computation
    double hybrid_time = hybridCPUGPUComputation(n / 4);
    
    // Multi-GPU unified memory (if available)
    multiGPUUnifiedMemory(n / 2);
    
    // Memory usage analysis
    analyzeMemoryUsage(n / 4);
    
    // Best practices summary
    printf("\n=== Unified Memory Best Practices ===\n");
    printf("1. Use memory hints (cudaMemAdvise) to guide migrations\n");
    printf("2. Prefetch data (cudaMemPrefetchAsync) to avoid page faults\n");
    printf("3. Consider access patterns - sequential is better than random\n");
    printf("4. Use concurrent managed access for better performance\n");
    printf("5. Be aware of page fault overhead on first access\n");
    printf("6. Monitor memory usage and migration patterns\n");
    
    printf("\nAdvantages of Unified Memory:\n");
    printf("+ Simplified programming model\n");
    printf("+ Automatic data migration\n");
    printf("+ CPU-GPU interoperability\n");
    printf("+ Multi-GPU support\n");
    
    printf("\nDisadvantages of Unified Memory:\n");
    printf("- Page fault overhead on first access\n");
    printf("- Less control over data movement\n");
    printf("- May not achieve peak performance for bandwidth-bound applications\n");
    printf("- Debugging can be more complex\n");
    
    printf("\nUnified Memory demonstration completed!\n");
    return 0;
}