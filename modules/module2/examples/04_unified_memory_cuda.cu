#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

// Simple kernel for unified memory demonstration
__global__ void vectorAdd(float *a, float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel that modifies data in-place
__global__ void vectorScale(float *data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// CPU function that processes the same data
void vectorScaleCPU(float *data, float scale, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

// Kernel for demonstrating data migration
__global__ void computeIntensive(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        
        // Compute-intensive operations
        for (int i = 0; i < 100; i++) {
            value = sinf(value) + cosf(value);
        }
        
        data[idx] = value;
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

class UnifiedMemoryDemo {
private:
    float *data;
    size_t size;
    size_t elements;
    
public:
    UnifiedMemoryDemo(size_t n) : elements(n), size(n * sizeof(float)) {
        // Allocate unified memory
        CUDA_CHECK(cudaMallocManaged(&data, size));
        
        // Initialize data on CPU
        printf("Initializing %zu elements in unified memory...\n", n);
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<float>(i % 1000);
        }
    }
    
    ~UnifiedMemoryDemo() {
        cudaFree(data);
    }
    
    void processOnGPU() {
        int blockSize = 256;
        int gridSize = (elements + blockSize - 1) / blockSize;
        
        printf("Processing on GPU...\n");
        
        // Data automatically migrates to GPU
        vectorScale<<<gridSize, blockSize>>>(data, 2.0f, elements);
        
        // Wait for completion
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void processOnCPU() {
        printf("Processing on CPU...\n");
        
        // Data automatically migrates back to CPU
        vectorScaleCPU(data, 0.5f, elements);
    }
    
    void printSample(int count = 10) {
        printf("First %d elements: ", count);
        for (int i = 0; i < count && i < elements; i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
    
    float* getData() { return data; }
    size_t getElements() { return elements; }
};

void demonstrateBasicUnifiedMemory() {
    printf("=== Basic Unified Memory Example ===\n");
    
    const size_t n = 1024 * 1024;  // 1M elements
    
    UnifiedMemoryDemo demo(n);
    
    printf("Initial data:\n");
    demo.printSample();
    
    // Process on GPU
    demo.processOnGPU();
    printf("After GPU processing (scale by 2.0):\n");
    demo.printSample();
    
    // Process on CPU
    demo.processOnCPU();
    printf("After CPU processing (scale by 0.5):\n");
    demo.printSample();
    
    printf("Basic unified memory demo completed.\n\n");
}

void compareUnifiedVsExplicit() {
    printf("=== Unified Memory vs Explicit Memory Management ===\n");
    
    const size_t n = 8 * 1024 * 1024;  // 8M elements
    const size_t bytes = n * sizeof(float);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test 1: Unified Memory
    float *unified_data;
    CUDA_CHECK(cudaMallocManaged(&unified_data, bytes));
    
    // Initialize
    for (size_t i = 0; i < n; i++) {
        unified_data[i] = static_cast<float>(i);
    }
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // GPU computation
    vectorScale<<<gridSize, blockSize>>>(unified_data, 2.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // CPU computation
    vectorScaleCPU(unified_data, 0.5f, n);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float unified_time;
    CUDA_CHECK(cudaEventElapsedTime(&unified_time, start, stop));
    
    // Test 2: Explicit Memory Management
    float *h_data = (float*)malloc(bytes);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize
    for (size_t i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i);
    }
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Explicit transfers and computation
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    vectorScale<<<gridSize, blockSize>>>(d_data, 2.0f, n);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    // CPU computation
    vectorScaleCPU(h_data, 0.5f, n);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float explicit_time;
    CUDA_CHECK(cudaEventElapsedTime(&explicit_time, start, stop));
    
    printf("Data size: %zu MB\n", bytes / (1024 * 1024));
    printf("Unified memory time: %.3f ms\n", unified_time);
    printf("Explicit memory time: %.3f ms\n", explicit_time);
    printf("Performance ratio: %.2fx %s\n", 
           fmax(unified_time, explicit_time) / fmin(unified_time, explicit_time),
           (unified_time < explicit_time) ? "(unified faster)" : "(explicit faster)");
    
    // Cleanup
    cudaFree(unified_data);
    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrateMemoryMigration() {
    printf("\n=== Memory Migration and Hints ===\n");
    
    const size_t n = 4 * 1024 * 1024;  // 4M elements
    const size_t bytes = n * sizeof(float);
    
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    
    // Initialize data
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<float>(i);
    }
    
    int device = 0;
    
    printf("Testing memory migration with prefetching and hints...\n");
    
    // Set memory advice
    CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device));
    CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device));
    
    // Prefetch to GPU
    printf("Prefetching to GPU...\n");
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, device));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // GPU computation (data already on GPU)
    CUDA_CHECK(cudaEventRecord(start));
    computeIntensive<<<gridSize, blockSize>>>(data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Prefetch to CPU
    printf("Prefetching to CPU...\n");
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // CPU computation (data already on CPU)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        float value = data[i];
        for (int j = 0; j < 10; j++) {  // Less intensive on CPU
            value = sinf(value) + cosf(value);
        }
        data[i] = value;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    printf("GPU computation time: %.3f ms\n", gpu_time);
    printf("CPU computation time: %.3f ms\n", cpu_time);
    
    // Test without prefetching for comparison
    printf("\nTesting without prefetching...\n");
    
    // Reset memory advice
    CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseUnsetReadMostly, device));
    CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseUnsetPreferredLocation, device));
    
    CUDA_CHECK(cudaEventRecord(start));
    computeIntensive<<<gridSize, blockSize>>>(data, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time_no_prefetch;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_no_prefetch, start, stop));
    
    printf("GPU time without prefetching: %.3f ms\n", gpu_time_no_prefetch);
    printf("Prefetching benefit: %.2fx speedup\n", gpu_time_no_prefetch / gpu_time);
    
    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrateMemoryPool() {
    printf("\n=== Memory Pool Management ===\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    if (!prop.memoryPoolsSupported) {
        printf("Memory pools not supported on this device.\n");
        return;
    }
    
    // Create memory pool
    cudaMemPool_t mempool;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = 0;
    
    CUDA_CHECK(cudaMemPoolCreate(&mempool, &poolProps));
    
    printf("Created memory pool for efficient allocation/deallocation.\n");
    
    // Allocate multiple arrays from pool
    std::vector<float*> arrays;
    const size_t array_size = 1024 * 1024 * sizeof(float);  // 1M floats each
    const int num_arrays = 10;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test pool allocation
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < num_arrays; i++) {
        float *ptr;
        CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, array_size, mempool, 0));
        arrays.push_back(ptr);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pool_alloc_time;
    CUDA_CHECK(cudaEventElapsedTime(&pool_alloc_time, start, stop));
    
    // Deallocate from pool
    CUDA_CHECK(cudaEventRecord(start));
    
    for (float *ptr : arrays) {
        CUDA_CHECK(cudaFreeAsync(ptr, 0));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pool_free_time;
    CUDA_CHECK(cudaEventElapsedTime(&pool_free_time, start, stop));
    
    // Compare with regular allocation
    arrays.clear();
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < num_arrays; i++) {
        float *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, array_size));
        arrays.push_back(ptr);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float regular_alloc_time;
    CUDA_CHECK(cudaEventElapsedTime(&regular_alloc_time, start, stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (float *ptr : arrays) {
        CUDA_CHECK(cudaFree(ptr));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float regular_free_time;
    CUDA_CHECK(cudaEventElapsedTime(&regular_free_time, start, stop));
    
    printf("Allocating %d arrays of %zu MB each:\n", num_arrays, array_size / (1024 * 1024));
    printf("Pool allocation time: %.3f ms\n", pool_alloc_time);
    printf("Regular allocation time: %.3f ms\n", regular_alloc_time);
    printf("Pool deallocation time: %.3f ms\n", pool_free_time);
    printf("Regular deallocation time: %.3f ms\n", regular_free_time);
    
    printf("Pool allocation speedup: %.2fx\n", regular_alloc_time / pool_alloc_time);
    printf("Pool deallocation speedup: %.2fx\n", regular_free_time / pool_free_time);
    
    CUDA_CHECK(cudaMemPoolDestroy(mempool));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Unified Memory Examples\n");
    printf("============================\n");
    
    // Check unified memory support
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Managed memory supported: %s\n", props.managedMemory ? "YES" : "NO");
    printf("Concurrent managed access: %s\n", props.concurrentManagedAccess ? "YES" : "NO");
    printf("Page migration supported: %s\n", props.pageableMemoryAccess ? "YES" : "NO");
    
    if (!props.managedMemory) {
        printf("Unified memory not supported on this device.\n");
        return 1;
    }
    
    // Run demonstrations
    demonstrateBasicUnifiedMemory();
    compareUnifiedVsExplicit();
    demonstrateMemoryMigration();
    demonstrateMemoryPool();
    
    // Educational summary
    printf("\n=== Unified Memory Guidelines ===\n");
    printf("✓ ADVANTAGES:\n");
    printf("  - Simplified memory management\n");
    printf("  - Automatic data migration between CPU and GPU\n");
    printf("  - Single address space for CPU and GPU\n");
    printf("  - Easier development and debugging\n");
    printf("  - Handles complex memory access patterns\n");
    
    printf("\n✓ BEST USE CASES:\n");
    printf("  - Prototyping and development\n");
    printf("  - Complex data structures with pointers\n");
    printf("  - Applications with unpredictable access patterns\n");
    printf("  - Sparse data processing\n");
    printf("  - Mixed CPU-GPU algorithms\n");
    
    printf("\n⚠ CONSIDERATIONS:\n");
    printf("  - Page fault overhead on first access\n");
    printf("  - May not achieve peak performance for simple patterns\n");
    printf("  - Limited by PCIe bandwidth for migrations\n");
    printf("  - Device-dependent behavior\n");
    
    printf("\n💡 OPTIMIZATION TIPS:\n");
    printf("  - Use cudaMemPrefetchAsync() to pre-migrate data\n");
    printf("  - Apply memory hints with cudaMemAdvise()\n");
    printf("  - Minimize CPU-GPU ping-ponging\n");
    printf("  - Consider access patterns in algorithm design\n");
    printf("  - Profile to identify migration hotspots\n");
    printf("  - Use memory pools for frequent allocations\n");
    
    printf("\nUnified memory examples completed successfully!\n");
    return 0;
}