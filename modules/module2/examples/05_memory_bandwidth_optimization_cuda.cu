#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

// Simple copy kernel (baseline)
__global__ void simpleCopy(float *input, float *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Vectorized copy using float4
__global__ void vectorizedCopy(float4 *input, float4 *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];  // 16 bytes in one transaction
    }
}

// Streaming copy with multiple elements per thread
__global__ void streamingCopy(float *input, float *output, size_t n, int elements_per_thread) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    for (int i = 0; i < elements_per_thread && base_idx + i < n; i++) {
        output[base_idx + i] = input[base_idx + i];
    }
}

// Memory bandwidth test kernel with different patterns
__global__ void stridedRead(float *input, float *output, size_t n, int stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t strided_idx = idx * stride;
    
    if (strided_idx < n) {
        output[idx] = input[strided_idx];
    }
}

// Reduction kernel optimized for memory bandwidth
__global__ void optimizedReduction(float *input, float *output, size_t n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load two elements per thread to improve bandwidth utilization
    sdata[tid] = 0;
    if (i < n) sdata[tid] += input[i];
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Transpose kernel optimized for memory bandwidth
__global__ void optimizedTranspose(float *input, float *output, int width, int height) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced read
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Coalesced write to transposed location
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Pinned memory demonstration
void* allocatePinnedMemory(size_t size) {
    void* ptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate pinned memory: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
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

class BandwidthBenchmark {
private:
    float *d_input, *d_output;
    float4 *d_input4, *d_output4;
    size_t size, elements;
    cudaEvent_t start, stop;
    
public:
    BandwidthBenchmark(size_t n) : elements(n), size(n * sizeof(float)) {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input, size));
        CUDA_CHECK(cudaMalloc(&d_output, size));
        CUDA_CHECK(cudaMalloc(&d_input4, size));
        CUDA_CHECK(cudaMalloc(&d_output4, size));
        
        // Create events
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Initialize data
        initializeData();
    }
    
    ~BandwidthBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_input4);
        cudaFree(d_output4);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void initializeData() {
        // Initialize with test pattern
        float *h_data = new float[elements];
        for (size_t i = 0; i < elements; i++) {
            h_data[i] = static_cast<float>(i % 1000);
        }
        
        CUDA_CHECK(cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input4, h_data, size, cudaMemcpyHostToDevice));
        
        delete[] h_data;
    }
    
    float measureKernel(void (*kernel)(void*, void*, size_t, int), 
                       void* input, void* output, size_t n, int param,
                       int blocks, int threads) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<blocks, threads>>>(input, output, n, param);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        return time;
    }
    
    double calculateBandwidth(float time_ms, size_t bytes_transferred) {
        return (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    }
    
    void runCopyBenchmarks() {
        printf("=== Memory Copy Benchmarks ===\n");
        
        const int threads = 256;
        const int blocks = (elements + threads - 1) / threads;
        const int blocks4 = (elements / 4 + threads - 1) / threads;
        
        // Simple copy
        CUDA_CHECK(cudaEventRecord(start));
        simpleCopy<<<blocks, threads>>>(d_input, d_output, elements);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float simple_time;
        CUDA_CHECK(cudaEventElapsedTime(&simple_time, start, stop));
        
        // Vectorized copy
        CUDA_CHECK(cudaEventRecord(start));
        vectorizedCopy<<<blocks4, threads>>>(d_input4, d_output4, elements / 4);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float vector_time;
        CUDA_CHECK(cudaEventElapsedTime(&vector_time, start, stop));
        
        // Streaming copy (2 elements per thread)
        const int elements_per_thread = 2;
        const int stream_blocks = (elements + threads * elements_per_thread - 1) / 
                                 (threads * elements_per_thread);
        
        CUDA_CHECK(cudaEventRecord(start));
        streamingCopy<<<stream_blocks, threads>>>(d_input, d_output, elements, elements_per_thread);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float stream_time;
        CUDA_CHECK(cudaEventElapsedTime(&stream_time, start, stop));
        
        size_t bytes_transferred = 2 * size;  // Read + write
        
        printf("Array size: %zu MB\n", size / (1024 * 1024));
        printf("Simple copy:     %.3f ms, %.2f GB/s\n", 
               simple_time, calculateBandwidth(simple_time, bytes_transferred));
        printf("Vectorized copy: %.3f ms, %.2f GB/s (%.2fx speedup)\n", 
               vector_time, calculateBandwidth(vector_time, bytes_transferred),
               simple_time / vector_time);
        printf("Streaming copy:  %.3f ms, %.2f GB/s (%.2fx speedup)\n", 
               stream_time, calculateBandwidth(stream_time, bytes_transferred),
               simple_time / stream_time);
    }
    
    void runStrideBenchmarks() {
        printf("\n=== Strided Access Benchmarks ===\n");
        
        const int threads = 256;
        int strides[] = {1, 2, 4, 8, 16, 32, 64};
        const int num_strides = sizeof(strides) / sizeof(strides[0]);
        
        printf("%-10s %10s %12s\n", "Stride", "Time (ms)", "Bandwidth (GB/s)");
        printf("----------------------------------------\n");
        
        for (int i = 0; i < num_strides; i++) {
            int stride = strides[i];
            int active_elements = elements / stride;
            int blocks = (active_elements + threads - 1) / threads;
            
            CUDA_CHECK(cudaEventRecord(start));
            stridedRead<<<blocks, threads>>>(d_input, d_output, elements, stride);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float time;
            CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
            
            size_t bytes_transferred = active_elements * sizeof(float) * 2;  // Read + write
            double bandwidth = calculateBandwidth(time, bytes_transferred);
            
            printf("%-10d %10.3f %12.2f\n", stride, time, bandwidth);
        }
    }
    
    void runReductionBenchmark() {
        printf("\n=== Optimized Reduction Benchmark ===\n");
        
        const int threads = 256;
        const int blocks = (elements + threads * 2 - 1) / (threads * 2);
        
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_result, blocks * sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        optimizedReduction<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_result, elements);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        
        // Copy result back and sum
        float *h_result = new float[blocks];
        CUDA_CHECK(cudaMemcpy(h_result, d_result, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        float final_result = 0;
        for (int i = 0; i < blocks; i++) {
            final_result += h_result[i];
        }
        
        size_t bytes_read = size;  // Only reading input
        double bandwidth = calculateBandwidth(time, bytes_read);
        
        printf("Reduction time: %.3f ms\n", time);
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
        printf("Result: %.2f\n", final_result);
        
        delete[] h_result;
        cudaFree(d_result);
    }
};

void compareMemoryTypes() {
    printf("\n=== Memory Type Comparison ===\n");
    
    const size_t n = 64 * 1024 * 1024;  // 64M elements
    const size_t bytes = n * sizeof(float);
    
    // Allocate different memory types
    float *h_pageable = (float*)malloc(bytes);
    float *h_pinned = (float*)allocatePinnedMemory(bytes);
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize data
    for (size_t i = 0; i < n; i++) {
        h_pageable[i] = static_cast<float>(i);
        h_pinned[i] = static_cast<float>(i);
    }
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test pageable memory transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pageable_time;
    CUDA_CHECK(cudaEventElapsedTime(&pageable_time, start, stop));
    
    // Test pinned memory transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float pinned_time;
    CUDA_CHECK(cudaEventElapsedTime(&pinned_time, start, stop));
    
    // Test async transfer with pinned memory
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float async_time;
    CUDA_CHECK(cudaEventElapsedTime(&async_time, start, stop));
    
    double data_gb = bytes / (1024.0 * 1024.0 * 1024.0);
    
    printf("Transfer size: %.2f GB\n", data_gb);
    printf("Pageable memory:   %.3f ms, %.2f GB/s\n", 
           pageable_time, data_gb / (pageable_time / 1000.0));
    printf("Pinned memory:     %.3f ms, %.2f GB/s (%.2fx speedup)\n", 
           pinned_time, data_gb / (pinned_time / 1000.0), pageable_time / pinned_time);
    printf("Async pinned:      %.3f ms, %.2f GB/s (%.2fx speedup)\n", 
           async_time, data_gb / (async_time / 1000.0), pageable_time / async_time);
    
    // Cleanup
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runTransposeBenchmark() {
    printf("\n=== Optimized Transpose Benchmark ===\n");
    
    const int width = 2048;
    const int height = 2048;
    const size_t size = width * height * sizeof(float);
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Initialize with pattern
    float *h_data = new float[width * height];
    for (int i = 0; i < width * height; i++) {
        h_data[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice));
    
    dim3 blockSize(32, 32);
    dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    optimizedTranspose<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    
    size_t bytes_transferred = 2 * size;  // Read + write
    double bandwidth = (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (time / 1000.0);
    
    printf("Matrix size: %dx%d\n", width, height);
    printf("Transpose time: %.3f ms\n", time);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    
    delete[] h_data;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Memory Bandwidth Optimization\n");
    printf("===================================\n");
    
    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    
    double theoretical_bandwidth = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical peak bandwidth: %.1f GB/s\n", theoretical_bandwidth);
    printf("Memory clock rate: %d MHz\n", props.memoryClockRate / 1000);
    printf("Memory bus width: %d bits\n", props.memoryBusWidth);
    printf("L2 cache size: %d MB\n", props.l2CacheSize / (1024 * 1024));
    
    // Run benchmarks
    const size_t test_elements = 32 * 1024 * 1024;  // 32M elements = 128MB
    BandwidthBenchmark benchmark(test_elements);
    
    benchmark.runCopyBenchmarks();
    benchmark.runStrideBenchmarks();
    benchmark.runReductionBenchmark();
    
    compareMemoryTypes();
    runTransposeBenchmark();
    
    // Educational summary
    printf("\n=== Memory Bandwidth Optimization Guidelines ===\n");
    printf("✓ MAXIMIZE BANDWIDTH:\n");
    printf("  - Use vectorized memory operations (float4, etc.)\n");
    printf("  - Ensure coalesced memory access patterns\n");
    printf("  - Use pinned memory for host-device transfers\n");
    printf("  - Overlap computation with memory transfers\n");
    printf("  - Minimize stride in memory access patterns\n");
    
    printf("\n✓ MEMORY HIERARCHY OPTIMIZATION:\n");
    printf("  - Leverage shared memory for data reuse\n");
    printf("  - Utilize L1/L2 cache effectively\n");
    printf("  - Consider memory bank conflicts\n");
    printf("  - Use appropriate block sizes for occupancy\n");
    
    printf("\n✓ ALGORITHMIC CONSIDERATIONS:\n");
    printf("  - Restructure algorithms for better locality\n");
    printf("  - Use tiling techniques for large datasets\n");
    printf("  - Consider memory-bound vs compute-bound trade-offs\n");
    printf("  - Implement streaming techniques for large data\n");
    
    printf("\n💡 PROFILING TIPS:\n");
    printf("  - Use nvprof/ncu to measure actual bandwidth utilization\n");
    printf("  - Monitor memory throughput vs peak bandwidth\n");
    printf("  - Identify memory access bottlenecks\n");
    printf("  - Compare achieved vs theoretical bandwidth\n");
    
    printf("\nMemory bandwidth optimization examples completed successfully!\n");
    return 0;
}