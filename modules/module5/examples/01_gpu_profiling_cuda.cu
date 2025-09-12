#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
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

// Custom profiling colors for NVTX
#define NVTX_COLOR_RED    0xFFFF0000
#define NVTX_COLOR_GREEN  0xFF00FF00
#define NVTX_COLOR_BLUE   0xFF0000FF
#define NVTX_COLOR_YELLOW 0xFFFFFF00

// Performance counter class for detailed analysis
class PerformanceProfiler {
private:
    std::map<std::string, std::vector<float>> timings;
    std::map<std::string, cudaEvent_t> startEvents;
    std::map<std::string, cudaEvent_t> stopEvents;
    
public:
    PerformanceProfiler() {
        // Initialize CUDA profiler
        cudaProfilerStart();
    }
    
    ~PerformanceProfiler() {
        // Cleanup events
        for (auto& pair : startEvents) {
            cudaEventDestroy(pair.second);
        }
        for (auto& pair : stopEvents) {
            cudaEventDestroy(pair.second);
        }
        cudaProfilerStop();
    }
    
    void startTimer(const std::string& name) {
        if (startEvents.find(name) == startEvents.end()) {
            cudaEventCreate(&startEvents[name]);
            cudaEventCreate(&stopEvents[name]);
        }
        cudaEventRecord(startEvents[name]);
    }
    
    void endTimer(const std::string& name) {
        cudaEventRecord(stopEvents[name]);
        cudaEventSynchronize(stopEvents[name]);
        
        float ms;
        cudaEventElapsedTime(&ms, startEvents[name], stopEvents[name]);
        timings[name].push_back(ms);
    }
    
    void printStatistics() {
        printf("\n=== Performance Profiling Statistics ===\n");
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel with different compute intensities for profiling analysis
__global__ void computeIntensiveKernel(float *data, int n, int iterations) {
    nvtxRangePushA("Compute Intensive Kernel");
    
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
    
    nvtxRangePop();
}

// Memory bandwidth intensive kernel
__global__ void memoryIntensiveKernel(float *input, float *output, int n) {
    nvtxRangePushA("Memory Intensive Kernel");
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Multiple memory accesses with minimal computation
        float sum = 0.0f;
        for (int offset = 0; offset < 8 && idx + offset < n; offset++) {
            sum += input[idx + offset];
        }
        output[idx] = sum / 8.0f;
    }
    
    nvtxRangePop();
}

// Kernel with poor memory coalescing for profiling analysis
__global__ void poorCoalescingKernel(float *data, int n, int stride) {
    nvtxRangePushA("Poor Coalescing Kernel");
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    
    if (strided_idx < n) {
        data[strided_idx] = data[strided_idx] * 2.0f + 1.0f;
    }
    
    nvtxRangePop();
}

// Optimized coalescing kernel for comparison
__global__ void goodCoalescingKernel(float *data, int n) {
    nvtxRangePushA("Good Coalescing Kernel");
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
    
    nvtxRangePop();
}

// Shared memory intensive kernel
__global__ void sharedMemoryKernel(float *input, float *output, int n) {
    nvtxRangePushA("Shared Memory Kernel");
    
    __shared__ float shared_data[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory
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
    
    nvtxRangePop();
}

// Divergent kernel for warp efficiency analysis
__global__ void divergentKernel(float *data, int n) {
    nvtxRangePushA("Divergent Kernel");
    
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
    
    nvtxRangePop();
}

// Optimized non-divergent kernel for comparison
__global__ void nonDivergentKernel(float *data, int n) {
    nvtxRangePushA("Non-Divergent Kernel");
    
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
    
    nvtxRangePop();
}

void analyzeDeviceProperties() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("=== Device Properties for Profiling Analysis ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Cores per MP: %d (estimated)\n", _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("Total Cores: %d (estimated)\n", prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("GPU Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.1f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per MP: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("\n");
}

// Helper function to convert SM version to core count (approximate)
int _ConvertSMVer2Cores(int major, int minor) {
    // Approximate cores per SM for different architectures
    int cores = 0;
    switch ((major << 4) + minor) {
        case 0x30: // Kepler
        case 0x32:
        case 0x35:
        case 0x37:
            cores = 192;
            break;
        case 0x50: // Maxwell
        case 0x52:
        case 0x53:
            cores = 128;
            break;
        case 0x60: // Pascal
        case 0x61:
        case 0x62:
            cores = 64;
            break;
        case 0x70: // Volta
        case 0x72:
        case 0x75: // Turing
            cores = 64;
            break;
        case 0x80: // Ampere
        case 0x86:
            cores = 64;
            break;
        case 0x90: // Hopper
            cores = 128;
            break;
        default:
            cores = 64; // Default estimate
    }
    return cores;
}

void runProfilingBenchmarks(PerformanceProfiler& profiler) {
    printf("=== Running Profiling Benchmarks ===\n");
    
    // Allocate memory
    float *h_data = new float[ARRAY_SIZE];
    float *h_output = new float[ARRAY_SIZE];
    float *d_data, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_data, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float)));
    
    // Initialize data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((ARRAY_SIZE + block.x - 1) / block.x);
    
    printf("Grid dimensions: %d blocks of %d threads\n", grid.x, block.x);
    printf("Total threads: %d\n", grid.x * block.x);
    printf("Array size: %d elements (%.1f MB)\n", ARRAY_SIZE, ARRAY_SIZE * sizeof(float) / 1024.0 / 1024.0);
    printf("\n");
    
    // Benchmark 1: Different compute intensities
    printf("1. Compute Intensity Analysis:\n");
    nvtxRangePushEx("Compute Intensity Analysis", NVTX_COLOR_RED);
    
    for (int intensity : {1, 10, 100}) {
        std::string name = "Compute_" + std::to_string(intensity) + "_iterations";
        
        for (int run = 0; run < NUM_ITERATIONS; run++) {
            profiler.startTimer(name);
            computeIntensiveKernel<<<grid, block>>>(d_data, ARRAY_SIZE, intensity);
            profiler.endTimer(name);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    nvtxRangePop();
    
    // Benchmark 2: Memory bandwidth analysis
    printf("2. Memory Bandwidth Analysis:\n");
    nvtxRangePushEx("Memory Bandwidth Analysis", NVTX_COLOR_GREEN);
    
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Memory_Intensive");
        memoryIntensiveKernel<<<grid, block>>>(d_data, d_output, ARRAY_SIZE);
        profiler.endTimer("Memory_Intensive");
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();
    
    // Benchmark 3: Memory coalescing comparison
    printf("3. Memory Coalescing Analysis:\n");
    nvtxRangePushEx("Memory Coalescing Analysis", NVTX_COLOR_BLUE);
    
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Good_Coalescing");
        goodCoalescingKernel<<<grid, block>>>(d_data, ARRAY_SIZE);
        profiler.endTimer("Good_Coalescing");
        CUDA_CHECK(cudaDeviceSynchronize());
        
        profiler.startTimer("Poor_Coalescing_Stride_8");
        poorCoalescingKernel<<<grid / 8, block>>>(d_data, ARRAY_SIZE, 8);
        profiler.endTimer("Poor_Coalescing_Stride_8");
        CUDA_CHECK(cudaDeviceSynchronize());
        
        profiler.startTimer("Poor_Coalescing_Stride_32");
        poorCoalescingKernel<<<grid / 32, block>>>(d_data, ARRAY_SIZE, 32);
        profiler.endTimer("Poor_Coalescing_Stride_32");
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();
    
    // Benchmark 4: Shared memory usage
    printf("4. Shared Memory Analysis:\n");
    nvtxRangePushEx("Shared Memory Analysis", NVTX_COLOR_YELLOW);
    
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Shared_Memory_Intensive");
        sharedMemoryKernel<<<grid, block>>>(d_data, d_output, ARRAY_SIZE);
        profiler.endTimer("Shared_Memory_Intensive");
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();
    
    // Benchmark 5: Warp divergence analysis
    printf("5. Warp Divergence Analysis:\n");
    nvtxRangePushEx("Warp Divergence Analysis", NVTX_COLOR_RED);
    
    for (int run = 0; run < NUM_ITERATIONS; run++) {
        profiler.startTimer("Divergent_Kernel");
        divergentKernel<<<grid, block>>>(d_data, ARRAY_SIZE);
        profiler.endTimer("Divergent_Kernel");
        CUDA_CHECK(cudaDeviceSynchronize());
        
        profiler.startTimer("Non_Divergent_Kernel");
        nonDivergentKernel<<<grid, block>>>(d_data, ARRAY_SIZE);
        profiler.endTimer("Non_Divergent_Kernel");
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_data;
    delete[] h_output;
}

void calculateTheoreticalLimits() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("=== Theoretical Performance Limits ===\n");
    
    // Memory bandwidth calculation
    double memoryBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // GB/s
    printf("Peak Memory Bandwidth: %.1f GB/s\n", memoryBandwidth);
    
    // Compute throughput estimation
    int coresPerSM = _ConvertSMVer2Cores(prop.major, prop.minor);
    int totalCores = prop.multiProcessorCount * coresPerSM;
    double computeThroughput = totalCores * prop.clockRate / 1e6; // GFLOPS (single precision)
    printf("Estimated Peak Compute (SP): %.1f GFLOPS\n", computeThroughput);
    
    // Roofline model breakpoint
    double arithmeticIntensity = computeThroughput / memoryBandwidth; // FLOPS/Byte
    printf("Compute/Memory Balance Point: %.1f FLOPS/Byte\n", arithmeticIntensity);
    
    printf("\nRoofline Model Interpretation:\n");
    printf("- Applications with <%.1f FLOPS/Byte are memory bound\n", arithmeticIntensity);
    printf("- Applications with >%.1f FLOPS/Byte are compute bound\n", arithmeticIntensity);
    printf("- Focus optimization efforts accordingly\n\n");
}

void printProfilingGuidance() {
    printf("=== Profiling Tool Usage Guide ===\n");
    printf("\nTo get detailed profiling data, use these commands:\n\n");
    
    printf("NVIDIA Nsight Compute (detailed kernel analysis):\n");
    printf("  ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed ./01_gpu_profiling_cuda\n");
    printf("  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy ./01_gpu_profiling_cuda\n");
    printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./01_gpu_profiling_cuda\n");
    printf("  ncu --set full ./01_gpu_profiling_cuda\n\n");
    
    printf("NVIDIA Nsight Systems (timeline analysis):\n");
    printf("  nsys profile --trace=cuda,nvtx,osrt --stats=true ./01_gpu_profiling_cuda\n");
    printf("  nsys profile --trace=cuda,nvtx --output=profile ./01_gpu_profiling_cuda\n\n");
    
    printf("Legacy nvprof (if available):\n");
    printf("  nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./01_gpu_profiling_cuda\n");
    printf("  nvprof --events gld_transactions,gst_transactions ./01_gpu_profiling_cuda\n\n");
    
    printf("Key metrics to analyze:\n");
    printf("- Memory throughput vs peak bandwidth\n");
    printf("- Achieved occupancy vs theoretical occupancy\n");
    printf("- Warp execution efficiency\n");
    printf("- Memory coalescing efficiency\n");
    printf("- Shared memory bank conflicts\n");
    printf("- Cache hit rates\n\n");
}

void analyzePerformanceResults(PerformanceProfiler& profiler) {
    // This would contain the analysis logic based on profiling results
    printf("=== Performance Analysis Insights ===\n");
    printf("Based on the timing results:\n\n");
    
    printf("Compute Intensity Analysis:\n");
    printf("- Higher iteration counts should show linear scaling if compute-bound\n");
    printf("- If scaling is sub-linear, you may be hitting memory bandwidth limits\n\n");
    
    printf("Memory Coalescing Analysis:\n");
    printf("- Good coalescing should show optimal memory bandwidth utilization\n");
    printf("- Poor coalescing (strided access) should show significant slowdown\n");
    printf("- Larger strides should perform worse than smaller strides\n\n");
    
    printf("Warp Divergence Analysis:\n");
    printf("- Divergent kernels should show lower performance due to serialized execution\n");
    printf("- Non-divergent implementations should utilize warps more efficiently\n\n");
    
    printf("Optimization Recommendations:\n");
    printf("1. Profile with detailed tools (ncu, nsys) to identify bottlenecks\n");
    printf("2. Focus on the largest performance gaps first\n");
    printf("3. Validate optimizations with before/after measurements\n");
    printf("4. Consider the roofline model for optimization strategy\n\n");
}

int main() {
    printf("=== GPU Profiling and Performance Analysis Demo ===\n\n");
    
    // Initialize profiler
    PerformanceProfiler profiler;
    
    // Analyze device properties
    analyzeDeviceProperties();
    
    // Calculate theoretical limits
    calculateTheoreticalLimits();
    
    // Run comprehensive benchmarks
    runProfilingBenchmarks(profiler);
    
    // Print statistics
    profiler.printStatistics();
    
    // Analyze results and provide guidance
    analyzePerformanceResults(profiler);
    
    // Print profiling tool usage guide
    printProfilingGuidance();
    
    printf("=== Profiling Demo Complete ===\n");
    printf("Use the suggested profiling commands above for detailed analysis!\n");
    
    return 0;
}