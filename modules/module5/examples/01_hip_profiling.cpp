#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#define ARRAY_SIZE (32 * 1024 * 1024)  // 32M elements
#define BLOCK_SIZE 256
#define NUM_ITERATIONS 10

// Performance counter class for detailed analysis
class HIPPerformanceProfiler {
private:
    std::map<std::string, std::vector<float>> timings;
    std::map<std::string, hipEvent_t> startEvents;
    std::map<std::string, hipEvent_t> stopEvents;
    
public:
    HIPPerformanceProfiler() {}
    
    ~HIPPerformanceProfiler() {
        // Cleanup events
        for (auto& pair : startEvents) {
            hipEventDestroy(pair.second);
        }
        for (auto& pair : stopEvents) {
            hipEventDestroy(pair.second);
        }
    }
    
    void startTimer(const std::string& name) {
        if (startEvents.find(name) == startEvents.end()) {
            hipEventCreate(&startEvents[name]);
            hipEventCreate(&stopEvents[name]);
        }
        hipEventRecord(startEvents[name]);
    }
    
    void endTimer(const std::string& name) {
        hipEventRecord(stopEvents[name]);
        hipEventSynchronize(stopEvents[name]);
        
        float ms;
        hipEventElapsedTime(&ms, startEvents[name], stopEvents[name]);
        timings[name].push_back(ms);
    }
    
    void printStatistics() {
        printf("\n=== HIP Performance Profiling Statistics ===\n");
        for (const auto& pair : timings) {
            const auto& times = pair.second;
            if (times.empty()) continue;
            
            float sum = 0.0f, min_time = times[0], max_time = times[0];
            for (float time : times) {
                sum += time;
                min_time = fminf(min_time, time);
                max_time = fmaxf(max_time, time);
            }
            
            float mean = sum / times.size();
            float variance = 0.0f;
            for (float time : times) {
                variance += (time - mean) * (time - mean);
            }
            float stddev = sqrtf(variance / times.size());
            
            printf("%-30s: %8.3f ± %6.3f ms (min: %6.3f, max: %6.3f, n=%zu)\n",
                   pair.first.c_str(), mean, stddev, min_time, max_time, times.size());
        }
    }
};

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel with different compute intensities for profiling analysis
__global__ void computeIntensiveKernel(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Variable compute intensity based on iterations
        for (int i = 0; i < iterations; i++) {
            value = sinf(value + i * 0.1f) * cosf(value - i * 0.1f);
            value += expf(value * 0.01f) * logf(fabsf(value) + 1.0f);
        }
        
        data[idx] = value;
    }
}

// Memory bandwidth intensive kernel
__global__ void memoryIntensiveKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Multiple memory accesses with minimal computation
        float sum = 0.0f;
        for (int offset = 0; offset < 8 && idx + offset < n; offset++) {
            sum += input[idx + offset];
        }
        output[idx] = sum / 8.0f;
    }
}

// Kernel with poor memory coalescing for profiling analysis
__global__ void poorCoalescingKernel(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    
    if (strided_idx < n) {
        data[strided_idx] = data[strided_idx] * 2.0f + 1.0f;
    }
}

// Optimized coalescing kernel for comparison
__global__ void goodCoalescingKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Shared memory intensive kernel (LDS in HIP/AMD terminology)
__global__ void sharedMemoryKernel(float *input, float *output, int n) {
    HIP_DYNAMIC_SHARED(float, shared_data)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory (Local Data Share on AMD)
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Perform computation using shared memory
    float result = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        result += shared_data[i] * shared_data[tid];
    }
    
    __syncthreads();
    
    // Write result
    if (idx < n) {
        output[idx] = result;
    }
}

// Divergent kernel for wavefront efficiency analysis
__global__ void divergentKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // High divergence - threads take different paths
        if (value > 0.5f) {
            for (int i = 0; i < 100; i++) {
                value = sinf(value + i);
            }
        } else if (value > 0.0f) {
            for (int i = 0; i < 50; i++) {
                value = cosf(value + i);
            }
        } else {
            for (int i = 0; i < 200; i++) {
                value = expf(value * 0.1f);
            }
        }
        
        data[idx] = value;
    }
}

// Optimized non-divergent kernel for comparison
__global__ void nonDivergentKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // All threads execute same operations, results selected by predicates
        float sin_result = sinf(value + 100.0f);
        float cos_result = cosf(value + 50.0f);
        float exp_result = expf(value * 0.1f + 200.0f);
        
        // Predicated selection instead of branching
        float result = (value > 0.5f) ? sin_result : 
                      (value > 0.0f) ? cos_result : exp_result;
        
        data[idx] = result;
    }
}

void analyzeDeviceProperties() {
    int device;
    hipGetDevice(&device);
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    
    printf("=== HIP Device Properties for Profiling Analysis ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (CUs): %d\n", prop.multiProcessorCount);
    printf("GPU Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    
    // Calculate theoretical memory bandwidth
    double memBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("Peak Memory Bandwidth: %.1f GB/s\n", memBandwidth);
    
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per CU: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp/Wavefront Size: %d\n", prop.warpSize);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    
    // AMD-specific information
    printf("\nAMD/HIP Specific Information:\n");
    printf("Architecture: %s\n", strstr(prop.name, "gfx") ? "RDNA/CDNA" : "Unknown");
    if (prop.warpSize == 64) {
        printf("Wavefront Size: 64 (typical AMD)\n");
    } else if (prop.warpSize == 32) {
        printf("Wavefront Size: 32 (RDNA2+ or NVIDIA via HIP)\n");
    }
    
    printf("\n");
}

void runHIPProfilingBenchmarks(HIPPerformanceProfiler& profiler) {
    printf("=== Running HIP Profiling Benchmarks ===\n");
    
    // Allocate memory
    float *h_data = new float[ARRAY_SIZE];
    float *h_output = new float[ARRAY_SIZE];
    float *d_data, *d_output;
    
    HIP_CHECK(hipMalloc(&d_data, ARRAY_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, ARRAY_SIZE * sizeof(float)));
    
    // Initialize data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    HIP_CHECK(hipMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(float), hipMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    
    printf("Grid dimensions: %d blocks of %d threads\n", grid.x, block.x);
    printf("Total threads: %d\n", grid.x * block.x);
    printf("Array size: %d elements (%.1f MB)\n", ARRAY_SIZE, ARRAY_SIZE * sizeof(float) / 1024.0 / 1024.0);
    printf("\n");
    
    // Benchmark 1: Different compute intensities
    printf("1. Compute Intensity Analysis:\n");
    for (int intensity : {1, 10, 100}) {
        std::string name = "Compute_" + std::to_string(intensity) + "_iterations";
        
        for (int run = 0; run < NUM_ITERATIONS; run++) {
            profiler.startTimer(name);
            hipLaunchKernelGGL(computeIntensiveKernel, grid, block, 0, 0, d_data, ARRAY_SIZE, intensity);
            profiler.endTimer(name);
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    
    // Benchmark 2: Memory bandwidth analysis
    printf("2. Memory Bandwidth Analysis:\n");
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Memory_Intensive");
        hipLaunchKernelGGL(memoryIntensiveKernel, grid, block, 0, 0, d_data, d_output, ARRAY_SIZE);
        profiler.endTimer("Memory_Intensive");
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Benchmark 3: Memory coalescing comparison
    printf("3. Memory Coalescing Analysis:\n");
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Good_Coalescing");
        hipLaunchKernelGGL(goodCoalescingKernel, grid, block, 0, 0, d_data, ARRAY_SIZE);
        profiler.endTimer("Good_Coalescing");
        HIP_CHECK(hipDeviceSynchronize());
        
        profiler.startTimer("Poor_Coalescing_Stride_8");
        hipLaunchKernelGGL(poorCoalescingKernel, dim3(grid.x / 8), block, 0, 0, d_data, ARRAY_SIZE, 8);
        profiler.endTimer("Poor_Coalescing_Stride_8");
        HIP_CHECK(hipDeviceSynchronize());
        
        profiler.startTimer("Poor_Coalescing_Stride_32");
        hipLaunchKernelGGL(poorCoalescingKernel, dim3(grid.x / 32), block, 0, 0, d_data, ARRAY_SIZE, 32);
        profiler.endTimer("Poor_Coalescing_Stride_32");
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Benchmark 4: Shared memory (LDS) usage
    printf("4. Local Data Share (LDS) Analysis:\n");
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Shared_Memory_Intensive");
        hipLaunchKernelGGL(sharedMemoryKernel, grid, block, BLOCK_SIZE * sizeof(float), 0, d_data, d_output, ARRAY_SIZE);
        profiler.endTimer("Shared_Memory_Intensive");
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Benchmark 5: Wavefront divergence analysis
    printf("5. Wavefront Divergence Analysis:\n");
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Divergent_Kernel");
        hipLaunchKernelGGL(divergentKernel, grid, block, 0, 0, d_data, ARRAY_SIZE);
        profiler.endTimer("Divergent_Kernel");
        HIP_CHECK(hipDeviceSynchronize());
        
        profiler.startTimer("Non_Divergent_Kernel");
        hipLaunchKernelGGL(nonDivergentKernel, grid, block, 0, 0, d_data, ARRAY_SIZE);
        profiler.endTimer("Non_Divergent_Kernel");
        HIP_CHECK(hipDeviceSynchronize());
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_output));
    delete[] h_data;
    delete[] h_output;
}

void calculateTheoreticalLimitsHIP() {
    int device;
    hipGetDevice(&device);
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    
    printf("=== Theoretical Performance Limits (HIP) ===\n");
    
    // Memory bandwidth calculation
    double memoryBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // GB/s
    printf("Peak Memory Bandwidth: %.1f GB/s\n", memoryBandwidth);
    
    // Estimate compute throughput (this is rough for different architectures)
    double computeThroughput = 0.0;
    if (strstr(prop.name, "gfx")) {
        // AMD GPU - rough estimate based on CUs and clock
        computeThroughput = prop.multiProcessorCount * 64 * prop.clockRate / 1e6; // Rough GFLOPS estimate
    } else {
        // NVIDIA GPU accessed via HIP
        computeThroughput = prop.multiProcessorCount * 64 * prop.clockRate / 1e6; // Rough GFLOPS estimate
    }
    
    printf("Estimated Peak Compute (SP): %.1f GFLOPS\n", computeThroughput);
    
    // Roofline model breakpoint
    if (computeThroughput > 0) {
        double arithmeticIntensity = computeThroughput / memoryBandwidth; // FLOPS/Byte
        printf("Compute/Memory Balance Point: %.1f FLOPS/Byte\n", arithmeticIntensity);
        
        printf("\nRoofline Model Interpretation:\n");
        printf("- Applications with <%.1f FLOPS/Byte are memory bound\n", arithmeticIntensity);
        printf("- Applications with >%.1f FLOPS/Byte are compute bound\n", arithmeticIntensity);
        printf("- Focus optimization efforts accordingly\n");
    }
    printf("\n");
}

void printHIPProfilingGuidance() {
    printf("=== HIP/ROCm Profiling Tool Usage Guide ===\n");
    printf("\nTo get detailed profiling data, use these commands:\n\n");
    
    printf("ROCm Profiler (rocprof):\n");
    printf("  rocprof --hip-trace --stats ./01_hip_profiling\n");
    printf("  rocprof --hsa-trace --stats ./01_hip_profiling\n");
    printf("  rocprof --input=metrics.txt ./01_hip_profiling\n");
    printf("  rocprof --timestamp on --hip-trace ./01_hip_profiling\n\n");
    
    printf("ROCTracer (low-level tracing):\n");
    printf("  roctracer --hip-trace ./01_hip_profiling\n");
    printf("  roctracer --hsa-trace ./01_hip_profiling\n\n");
    
    printf("If using HIP on NVIDIA (HIP can also profile NVIDIA GPUs):\n");
    printf("  nsys profile --trace=cuda,hip --stats=true ./01_hip_profiling\n");
    printf("  ncu --hip-trace ./01_hip_profiling\n\n");
    
    printf("Key HIP/AMD metrics to analyze:\n");
    printf("- Memory throughput vs peak bandwidth\n");
    printf("- Wavefront occupancy (similar to CUDA occupancy)\n");
    printf("- Wavefront execution efficiency\n");
    printf("- Memory coalescing efficiency\n");
    printf("- LDS (Local Data Share) bank conflicts\n");
    printf("- Cache hit rates\n");
    printf("- Kernel execution time\n\n");
    
    printf("AMD-specific considerations:\n");
    printf("- Wavefront size (typically 64, but 32 on some newer architectures)\n");
    printf("- GCN/RDNA/CDNA architecture differences\n");
    printf("- Memory hierarchy differences vs NVIDIA\n");
    printf("- ROCm software stack optimizations\n\n");
}

void analyzeHIPPerformanceResults(HIPPerformanceProfiler& profiler) {
    printf("=== HIP Performance Analysis Insights ===\n");
    printf("Based on the timing results:\n\n");
    
    printf("Compute Intensity Analysis:\n");
    printf("- Higher iteration counts should show linear scaling if compute-bound\n");
    printf("- AMD GPUs may have different scaling characteristics than NVIDIA\n");
    printf("- Consider wavefront size (64 vs 32) impact on performance\n\n");
    
    printf("Memory Coalescing Analysis:\n");
    printf("- HIP memory coalescing follows similar patterns to CUDA\n");
    printf("- AMD memory subsystem may have different optimal patterns\n");
    printf("- Check for architecture-specific memory access optimizations\n\n");
    
    printf("Wavefront Divergence Analysis:\n");
    printf("- Similar to CUDA warps, but with potentially different wavefront sizes\n");
    printf("- AMD GCN/RDNA architectures handle divergence differently than NVIDIA\n");
    printf("- Consider architecture-specific branching optimizations\n\n");
    
    printf("HIP Optimization Recommendations:\n");
    printf("1. Profile with ROCm tools (rocprof, roctracer) for AMD GPUs\n");
    printf("2. Consider both AMD and NVIDIA optimization when using HIP\n");
    printf("3. Test performance on target hardware architectures\n");
    printf("4. Use HIP's portability features for cross-vendor optimization\n");
    printf("5. Consider memory hierarchy differences between vendors\n\n");
}

int main() {
    printf("=== HIP GPU Profiling and Performance Analysis Demo ===\n\n");
    
    // Initialize profiler
    HIPPerformanceProfiler profiler;
    
    // Analyze device properties
    analyzeDeviceProperties();
    
    // Calculate theoretical limits
    calculateTheoreticalLimitsHIP();
    
    // Run comprehensive benchmarks
    runHIPProfilingBenchmarks(profiler);
    
    // Print statistics
    profiler.printStatistics();
    
    // Analyze results and provide guidance
    analyzeHIPPerformanceResults(profiler);
    
    // Print profiling tool usage guide
    printHIPProfilingGuidance();
    
    printf("=== HIP Profiling Demo Complete ===\n");
    printf("Use the suggested profiling commands above for detailed analysis!\n");
    printf("Remember: HIP provides portability across AMD and NVIDIA GPUs.\n");
    
    return 0;
}