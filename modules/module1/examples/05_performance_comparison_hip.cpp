#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

// CPU version of vector addition
void addVectorsCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// HIP GPU kernel
__global__ void addVectorsGPU(float *a, float *b, float *c, int n) {
    int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// HIP timer class using HIP events
class HipTimer {
    hipEvent_t start, stop;
    float elapsedTime;
public:
    HipTimer() {
        hipEventCreate(&start);
        hipEventCreate(&stop);
    }
    
    void startTimer() {
        hipEventRecord(start, 0);
    }
    
    void stopTimer() {
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
    }
    
    float getElapsedMs() { return elapsedTime; }
    
    ~HipTimer() {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
};

// CPU timer class
class CpuTimer {
    std::chrono::high_resolution_clock::time_point start, end;
public:
    void startTimer() { 
        start = std::chrono::high_resolution_clock::now(); 
    }
    
    void stopTimer() { 
        end = std::chrono::high_resolution_clock::now(); 
    }
    
    double getElapsedMs() {
        return std::chrono::duration<double, std::milli>(end - start).count();
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

int main() {
    // Get device information
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    printf("Performance Comparison: CPU vs HIP GPU Vector Addition\n");
    printf("=====================================================\n");
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
#else
    printf("Unknown\n");
#endif
    printf("\n");
    
    // Test different vector sizes
    int sizes[] = {1024, 10240, 102400, 1024000, 10240000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Vector Size\tCPU Time (ms)\tGPU Time (ms)\tSpeedup\tBandwidth (GB/s)\tEfficiency\n");
    printf("----------\t------------\t------------\t-------\t----------------\t----------\n");
    
    for (int test = 0; test < num_sizes; test++) {
        const int N = sizes[test];
        const int bytes = N * sizeof(float);
        
        // Host vectors
        float *h_a = (float*)malloc(bytes);
        float *h_b = (float*)malloc(bytes);
        float *h_c_cpu = (float*)malloc(bytes);
        float *h_c_gpu = (float*)malloc(bytes);
        
        // Initialize input vectors
        for (int i = 0; i < N; i++) {
            h_a[i] = sin(i) * sin(i);
            h_b[i] = cos(i) * cos(i);
        }
        
        // CPU benchmark
        CpuTimer cpu_timer;
        cpu_timer.startTimer();
        addVectorsCPU(h_a, h_b, h_c_cpu, N);
        cpu_timer.stopTimer();
        double cpu_time = cpu_timer.getElapsedMs();
        
        // GPU setup
        float *d_a, *d_b, *d_c;
        HIP_CHECK(hipMalloc(&d_a, bytes));
        HIP_CHECK(hipMalloc(&d_b, bytes));
        HIP_CHECK(hipMalloc(&d_c, bytes));
        
        // Copy data to GPU
        HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
        
        // GPU benchmark (kernel only)
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        HipTimer gpu_timer;
        gpu_timer.startTimer();
        addVectorsGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        gpu_timer.stopTimer();
        float gpu_time = gpu_timer.getElapsedMs();
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_c_gpu, d_c, bytes, hipMemcpyDeviceToHost));
        
        // Verify results match
        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
                correct = false;
                break;
            }
        }
        
        // Calculate performance metrics
        double speedup = cpu_time / gpu_time;
        double bandwidth = (3.0 * bytes / (1024.0 * 1024.0 * 1024.0)) / (gpu_time / 1000.0); // GB/s
        
        // Calculate theoretical peak bandwidth and efficiency
        double theoretical_bandwidth = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6;
        double efficiency = (bandwidth / theoretical_bandwidth) * 100.0;
        
        printf("%d\t\t%.3f\t\t%.3f\t\t%.2fx\t%.2f\t\t%.1f%%\n", 
               N, cpu_time, gpu_time, speedup, bandwidth, efficiency);
        
        if (!correct) {
            printf("ERROR: Results don't match for size %d\n", N);
        }
        
        // Cleanup
        free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
        hipFree(d_a); hipFree(d_b); hipFree(d_c);
    }
    
    // Additional GPU information
    printf("\n=== Device Performance Characteristics ===\n");
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
    printf("Memory Clock Rate: %.2f GHz\n", props.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    printf("Multiprocessor Count: %d\n", props.multiProcessorCount);
    printf("Max Threads per Multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Total Compute Units: %d\n", props.multiProcessorCount);
    
    // Memory bandwidth test with different access patterns
    printf("\n=== Memory Access Pattern Analysis ===\n");
    const int test_size = 1024 * 1024; // 1M elements
    const int test_bytes = test_size * sizeof(float);
    
    float *d_test_a, *d_test_b, *d_test_c;
    HIP_CHECK(hipMalloc(&d_test_a, test_bytes));
    HIP_CHECK(hipMalloc(&d_test_b, test_bytes));
    HIP_CHECK(hipMalloc(&d_test_c, test_bytes));
    
    // Test different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    printf("Block Size\tTime (ms)\tBandwidth (GB/s)\tOccupancy\n");
    printf("---------\t--------\t---------------\t---------\n");
    
    for (int i = 0; i < num_block_sizes; i++) {
        int blockSize = block_sizes[i];
        if (blockSize > props.maxThreadsPerBlock) continue;
        
        int gridSize = (test_size + blockSize - 1) / blockSize;
        
        // Calculate occupancy
        int maxActiveBlocks;
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, addVectorsGPU, blockSize, 0);
        float occupancy = (maxActiveBlocks * blockSize / (float)props.maxThreadsPerMultiProcessor) * 100.0f;
        
        HipTimer timer;
        timer.startTimer();
        addVectorsGPU<<<gridSize, blockSize>>>(d_test_a, d_test_b, d_test_c, test_size);
        timer.stopTimer();
        
        float time_ms = timer.getElapsedMs();
        double bandwidth = (3.0 * test_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
        
        printf("%d\t\t%.3f\t\t%.2f\t\t%.1f%%\n", blockSize, time_ms, bandwidth, occupancy);
    }
    
    hipFree(d_test_a); hipFree(d_test_b); hipFree(d_test_c);
    
    // Suggest optimal configuration
    int optimalBlockSize;
    int minGridSize;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, addVectorsGPU, 0, 0));
    printf("\nRecommended block size for this kernel: %d\n", optimalBlockSize);
    
    return 0;
}