#include <cuda_runtime.h>
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

// GPU kernel
__global__ void addVectorsGPU(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// GPU timer class using CUDA events
class GpuTimer {
    cudaEvent_t start, stop;
    float elapsedTime;
public:
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    void startTimer() {
        cudaEventRecord(start, 0);
    }
    
    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }
    
    float getElapsedMs() { return elapsedTime; }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Test different vector sizes
    int sizes[] = {1024, 10240, 102400, 1024000, 10240000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Performance Comparison: CPU vs GPU Vector Addition\n");
    printf("===================================================\n");
    printf("Vector Size\tCPU Time (ms)\tGPU Time (ms)\tSpeedup\tBandwidth (GB/s)\n");
    printf("----------\t------------\t------------\t-------\t----------------\n");
    
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
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        
        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        
        // GPU benchmark (kernel only)
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        GpuTimer gpu_timer;
        gpu_timer.startTimer();
        addVectorsGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        gpu_timer.stopTimer();
        float gpu_time = gpu_timer.getElapsedMs();
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
        
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
        
        printf("%d\t\t%.3f\t\t%.3f\t\t%.2fx\t%.2f\n", 
               N, cpu_time, gpu_time, speedup, bandwidth);
        
        if (!correct) {
            printf("ERROR: Results don't match for size %d\n", N);
        }
        
        // Cleanup
        free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    
    // Additional GPU information
    printf("\n");
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
    
    return 0;
}