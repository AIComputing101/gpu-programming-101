#include <hip/hip_runtime.h>
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

// AMD-optimized vectorized copy with wavefront awareness
__global__ void amdVectorizedCopy(float4 *input, float4 *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 64; // AMD wavefront size
    
    if (idx < n) {
        // Process multiple elements per wavefront for better memory utilization
        float4 data = input[idx];
        
        // AMD GPU optimization: ensure 64-byte aligned accesses
        if (lane_id == 0) {
            // Wavefront leader can do additional prefetching
            if (idx + 16 < n) {
                __builtin_prefetch(&input[idx + 16], 0, 3);
            }
        }
        
        output[idx] = data;
    }
}

// Streaming copy with multiple elements per thread
__global__ void streamingCopy(float *input, float *output, size_t n, int elements_per_thread) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    for (int i = 0; i < elements_per_thread && base_idx + i < n; i++) {
        output[base_idx + i] = input[base_idx + i];
    }
}

// AMD-optimized streaming copy with cache-line awareness
__global__ void amdStreamingCopy(float *input, float *output, size_t n, int elements_per_thread) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    
    // AMD GPU optimization: process in 64-byte (16 float) chunks for cache efficiency
    const int cache_line_elements = 16;
    
    for (int i = 0; i < elements_per_thread && base_idx + i < n; i += cache_line_elements) {
        // Process up to cache line worth of data
        int chunk_size = min(cache_line_elements, elements_per_thread - i);
        chunk_size = min(chunk_size, (int)(n - (base_idx + i)));
        
        for (int j = 0; j < chunk_size; j++) {
            output[base_idx + i + j] = input[base_idx + i + j];
        }
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
    __shared__ float sdata[256];  // Using explicit size for AMD compatibility
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load two elements per thread to improve bandwidth utilization
    sdata[tid] = 0;
    if (i < n) sdata[tid] += input[i];
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();
    
    // Reduction in shared memory with AMD optimization
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp/wavefront reduction without synchronization
    if (tid < 32) {
        volatile float* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// AMD-optimized reduction using wavefront primitives
__global__ void amdWavefrontReduction(float *input, float *output, size_t n) {
    __shared__ float sdata[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;
    
    // Load data
    float value = 0;
    if (i < n) value += input[i];
    if (i + blockDim.x < n) value += input[i + blockDim.x];
    
    // Wavefront reduction using shuffle operations
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    
    // Store wavefront results in shared memory
    if (lane_id == 0) {
        sdata[warp_id] = value;
    }
    
    __syncthreads();
    
    // Final reduction of wavefront results
    if (warp_id == 0) {
        value = (lane_id < (blockDim.x / 64)) ? sdata[lane_id] : 0;
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            value += __shfl_down(value, offset);
        }
        
        if (lane_id == 0) {
            output[blockIdx.x] = value;
        }
    }
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
    
    // Calculate transposed coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Coalesced write
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// AMD-optimized transpose with LDS bank conflict avoidance
__global__ void amdOptimizedTranspose(float *input, float *output, int width, int height) {
    // AMD LDS has 32 banks, so we pad to avoid bank conflicts
    __shared__ float tile[32][32 + 1];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced read with 64-byte cache line optimization
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Coalesced write optimized for AMD memory hierarchy
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
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

class BandwidthTester {
private:
    size_t size;
    size_t elements;
    float *d_input, *d_output;
    
public:
    BandwidthTester(size_t n) : elements(n), size(n * sizeof(float)) {
        HIP_CHECK(hipMalloc(&d_input, size));
        HIP_CHECK(hipMalloc(&d_output, size));
        
        // Initialize input data
        float *h_input = (float*)malloc(size);
        for (size_t i = 0; i < n; i++) {
            h_input[i] = static_cast<float>(i % 1000);
        }
        HIP_CHECK(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
        free(h_input);
    }
    
    ~BandwidthTester() {
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
    
    double testBandwidth(const char* test_name, void (*kernel)(float*, float*, size_t), 
                        int blockSize = 256, int elementsPerThread = 1) {
        int gridSize = (elements + blockSize * elementsPerThread - 1) / (blockSize * elementsPerThread);
        
        // Warm up
        for (int i = 0; i < 3; i++) {
            hipLaunchKernelGGL((void*)kernel, gridSize, blockSize, 0, 0, 
                              d_input, d_output, elements);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        const int num_iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < num_iterations; i++) {
            hipLaunchKernelGGL((void*)kernel, gridSize, blockSize, 0, 0, 
                              d_input, d_output, elements);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float milliseconds;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        
        // Calculate bandwidth (read + write)
        double bytes_transferred = 2.0 * size * num_iterations;
        double bandwidth_gb_s = bytes_transferred / (milliseconds * 1e6);
        
        printf("%-30s: %8.2f GB/s (%6.3f ms)\n", test_name, bandwidth_gb_s, milliseconds / num_iterations);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        
        return bandwidth_gb_s;
    }
    
    double testVectorizedBandwidth(const char* test_name, 
                                  void (*kernel)(float4*, float4*, size_t)) {
        int blockSize = 256;
        int gridSize = (elements / 4 + blockSize - 1) / blockSize;
        
        // Warm up
        for (int i = 0; i < 3; i++) {
            hipLaunchKernelGGL((void*)kernel, gridSize, blockSize, 0, 0, 
                              (float4*)d_input, (float4*)d_output, elements / 4);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        const int num_iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < num_iterations; i++) {
            hipLaunchKernelGGL((void*)kernel, gridSize, blockSize, 0, 0, 
                              (float4*)d_input, (float4*)d_output, elements / 4);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float milliseconds;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        
        double bytes_transferred = 2.0 * size * num_iterations;
        double bandwidth_gb_s = bytes_transferred / (milliseconds * 1e6);
        
        printf("%-30s: %8.2f GB/s (%6.3f ms)\n", test_name, bandwidth_gb_s, milliseconds / num_iterations);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        
        return bandwidth_gb_s;
    }
};

void measureTheoreticalBandwidth() {
    printf("\n=== Theoretical Bandwidth Analysis ===\n");
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    
    // Calculate theoretical bandwidth
    double memory_clock_ghz = prop.memoryClockRate / 1e6;
    double theoretical_bandwidth = 2.0 * memory_clock_ghz * (prop.memoryBusWidth / 8.0);
    
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoretical_bandwidth);
    
#ifdef __HIP_PLATFORM_AMD__
    printf("\nAMD GPU Memory Characteristics:\n");
    printf("- HBM2/HBM3 memory technology\n");
    printf("- Wide memory bus (typically 4096-bit)\n");
    printf("- High bandwidth, lower latency than GDDR\n");
    printf("- 64-byte cache line size\n");
    printf("- NUMA-aware memory access\n");
#endif
}

void analyzeAccessPatterns() {
    printf("\n=== Memory Access Pattern Analysis ===\n");
    
    const size_t n = 32 * 1024 * 1024; // 32M elements (128 MB)
    BandwidthTester tester(n);
    
    printf("Testing different access patterns:\n");
    
    // Test stride patterns
    float *d_temp;
    HIP_CHECK(hipMalloc(&d_temp, n * sizeof(float)));
    
    for (int stride = 1; stride <= 16; stride *= 2) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        int blockSize = 256;
        int gridSize = (n / stride + blockSize - 1) / blockSize;
        
        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(stridedRead, gridSize, blockSize, 0, 0, 
                          tester.d_input, d_temp, n, stride);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float milliseconds;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        
        double bytes_read = (n / stride) * sizeof(float);
        double bandwidth = bytes_read / (milliseconds * 1e6);
        
        printf("Stride %2d: %8.2f GB/s\n", stride, bandwidth);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    HIP_CHECK(hipFree(d_temp));
}

void demonstrateBandwidthOptimization() {
    printf("=== HIP Memory Bandwidth Optimization Demo ===\n");
    
    measureTheoreticalBandwidth();
    
    const size_t n = 64 * 1024 * 1024; // 64M elements (256 MB)
    BandwidthTester tester(n);
    
    printf("\n=== Copy Kernel Performance ===\n");
    
    // Test different copy strategies
    tester.testBandwidth("Simple Copy", simpleCopy);
    tester.testVectorizedBandwidth("Vectorized Copy (float4)", vectorizedCopy);
    
#ifdef __HIP_PLATFORM_AMD__
    tester.testVectorizedBandwidth("AMD Optimized Copy", amdVectorizedCopy);
#endif
    
    // Test streaming with different elements per thread
    printf("\n=== Streaming Copy Performance ===\n");
    for (int ept = 1; ept <= 8; ept *= 2) {
        char name[64];
        snprintf(name, sizeof(name), "Streaming Copy (%d elem/thread)", ept);
        
        int blockSize = 256;
        int gridSize = (n + blockSize * ept - 1) / (blockSize * ept);
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        const int num_iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < num_iterations; i++) {
            hipLaunchKernelGGL(streamingCopy, gridSize, blockSize, 0, 0, 
                              tester.d_input, tester.d_output, n, ept);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float milliseconds;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        
        double bytes_transferred = 2.0 * n * sizeof(float) * num_iterations;
        double bandwidth_gb_s = bytes_transferred / (milliseconds * 1e6);
        
        printf("%-30s: %8.2f GB/s\n", name, bandwidth_gb_s);
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    analyzeAccessPatterns();
    
    printf("\n=== Optimization Guidelines ===\n");
    printf("1. Use vectorized loads/stores (float4, float2)\n");
    printf("2. Ensure memory accesses are coalesced\n");
    printf("3. Process multiple elements per thread\n");
    printf("4. Align data to cache line boundaries\n");
    printf("5. Minimize stride patterns in memory access\n");
    
#ifdef __HIP_PLATFORM_AMD__
    printf("\nAMD-Specific Optimizations:\n");
    printf("- Optimize for 64-byte cache lines\n");
    printf("- Use wavefront-aware access patterns\n");
    printf("- Consider NUMA topology for multi-GPU\n");
    printf("- Leverage HBM bandwidth characteristics\n");
#endif
}

int main() {
    printf("HIP Memory Bandwidth Optimization Example\n");
    printf("=========================================\n");
    
    demonstrateBandwidthOptimization();
    
    return 0;
}