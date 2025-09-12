#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <vector>
#include <algorithm>

#define VECTOR_SIZE (32 * 1024 * 1024)  // 32M elements
#define MATRIX_SIZE 2048

// Matrix multiplication kernel
__global__ void matrixMul(float *A, float *B, float *C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Vector reduction kernel
__global__ void vectorReduction(float *input, float *output, int n) {
    HIP_DYNAMIC_SHARED(float, shared)
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    shared[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Simple compute kernel for load balancing tests
__global__ void computeWorkload(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < iterations; i++) {
            value = sinf(value + i) * cosf(value - i);
        }
        data[idx] = value;
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

struct GPUInfo {
    int deviceId;
    hipDeviceProp_t properties;
    size_t freeMemory;
    size_t totalMemory;
    double computeCapabilityScore;
};

class MultiGPUManager {
private:
    std::vector<GPUInfo> gpus;
    std::vector<hipStream_t> streams;
    int numGPUs;
    
public:
    MultiGPUManager() {
        HIP_CHECK(hipGetDeviceCount(&numGPUs));
        gpus.resize(numGPUs);
        streams.resize(numGPUs);
        
        printf("=== Multi-GPU System Detection ===\n");
        printf("Found %d GPU(s)\n\n", numGPUs);
        
        // Initialize GPU information
        for (int i = 0; i < numGPUs; i++) {
            HIP_CHECK(hipSetDevice(i));
            
            gpus[i].deviceId = i;
            HIP_CHECK(hipGetDeviceProperties(&gpus[i].properties, i));
            HIP_CHECK(hipMemGetInfo(&gpus[i].freeMemory, &gpus[i].totalMemory));
            
            // Simple compute capability score
            gpus[i].computeCapabilityScore = gpus[i].properties.multiProcessorCount * 
                                           gpus[i].properties.maxThreadsPerBlock / 1000.0;
            
            // Create stream for this GPU
            HIP_CHECK(hipStreamCreate(&streams[i]));
            
            printf("GPU %d: %s\n", i, gpus[i].properties.name);
            printf("  Compute Units: %d\n", gpus[i].properties.multiProcessorCount);
            printf("  Global Memory: %.1f GB (%.1f GB free)\n", 
                   gpus[i].totalMemory / (1024.0f*1024.0f*1024.0f),
                   gpus[i].freeMemory / (1024.0f*1024.0f*1024.0f));
            printf("  Max Threads per Block: %d\n", gpus[i].properties.maxThreadsPerBlock);
            printf("  Warp Size: %d\n", gpus[i].properties.warpSize);
            printf("  Compute Capability Score: %.1f\n", gpus[i].computeCapabilityScore);
            printf("\n");
        }
    }
    
    ~MultiGPUManager() {
        for (int i = 0; i < numGPUs; i++) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
    }
    
    int getNumGPUs() const { return numGPUs; }
    const GPUInfo& getGPU(int id) const { return gpus[id]; }
    hipStream_t getStream(int id) const { return streams[id]; }
    
    // Calculate load distribution based on GPU capabilities
    std::vector<int> calculateLoadDistribution(int totalWork) {
        std::vector<int> distribution(numGPUs, 0);
        
        if (numGPUs == 1) {
            distribution[0] = totalWork;
            return distribution;
        }
        
        // Calculate total compute capability
        double totalCapability = 0;
        for (const auto& gpu : gpus) {
            totalCapability += gpu.computeCapabilityScore;
        }
        
        // Distribute work proportionally
        int distributedWork = 0;
        for (int i = 0; i < numGPUs - 1; i++) {
            distribution[i] = (int)(totalWork * gpus[i].computeCapabilityScore / totalCapability);
            distributedWork += distribution[i];
        }
        
        // Assign remaining work to last GPU
        distribution[numGPUs - 1] = totalWork - distributedWork;
        
        printf("Load distribution:\n");
        for (int i = 0; i < numGPUs; i++) {
            printf("  GPU %d: %d elements (%.1f%%)\n", 
                   i, distribution[i], 100.0 * distribution[i] / totalWork);
        }
        printf("\n");
        
        return distribution;
    }
};

// Single GPU baseline
double runSingleGPU(float *h_data, int totalSize) {
    printf("Running single GPU baseline...\n");
    
    HIP_CHECK(hipSetDevice(0));
    
    float *d_data;
    size_t bytes = totalSize * sizeof(float);
    
    HIP_CHECK(hipMalloc(&d_data, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    HIP_CHECK(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((totalSize + block.x - 1) / block.x);
    hipLaunchKernelGGL(computeWorkload, grid, block, 0, 0, d_data, totalSize, 1000);
    
    HIP_CHECK(hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    HIP_CHECK(hipFree(d_data));
    
    double timeMs = duration.count() / 1000.0;
    printf("Single GPU time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Equal distribution multi-GPU
double runMultiGPUEqual(MultiGPUManager& manager, float *h_data, int totalSize) {
    printf("Running multi-GPU with equal distribution...\n");
    
    int numGPUs = manager.getNumGPUs();
    int chunkSize = totalSize / numGPUs;
    
    std::vector<float*> d_data(numGPUs);
    std::vector<size_t> chunkSizes(numGPUs);
    
    // Calculate chunk sizes
    for (int i = 0; i < numGPUs - 1; i++) {
        chunkSizes[i] = chunkSize;
    }
    chunkSizes[numGPUs - 1] = totalSize - chunkSize * (numGPUs - 1);
    
    // Allocate device memory
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&d_data[i], chunkSizes[i] * sizeof(float)));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parallel execution using OpenMP
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        HIP_CHECK(hipSetDevice(gpu));
        
        int offset = gpu * chunkSize;
        size_t chunkBytes = chunkSizes[gpu] * sizeof(float);
        
        // Transfer data to GPU
        HIP_CHECK(hipMemcpyAsync(d_data[gpu], h_data + offset, chunkBytes,
                                hipMemcpyHostToDevice, manager.getStream(gpu)));
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((chunkSizes[gpu] + block.x - 1) / block.x);
        hipLaunchKernelGGL(computeWorkload, grid, block, 0, manager.getStream(gpu),
                          d_data[gpu], chunkSizes[gpu], 1000);
        
        // Transfer results back
        HIP_CHECK(hipMemcpyAsync(h_data + offset, d_data[gpu], chunkBytes,
                                hipMemcpyDeviceToHost, manager.getStream(gpu)));
    }
    
    // Wait for all GPUs to complete
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(manager.getStream(i)));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(d_data[i]));
    }
    
    double timeMs = duration.count() / 1000.0;
    printf("Multi-GPU equal distribution time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Weighted distribution multi-GPU
double runMultiGPUWeighted(MultiGPUManager& manager, float *h_data, int totalSize) {
    printf("Running multi-GPU with weighted distribution...\n");
    
    int numGPUs = manager.getNumGPUs();
    std::vector<int> distribution = manager.calculateLoadDistribution(totalSize);
    
    std::vector<float*> d_data(numGPUs);
    
    // Allocate device memory
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        if (distribution[i] > 0) {
            HIP_CHECK(hipMalloc(&d_data[i], distribution[i] * sizeof(float)));
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parallel execution with weighted distribution
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        if (distribution[gpu] == 0) continue;
        
        HIP_CHECK(hipSetDevice(gpu));
        
        // Calculate offset
        int offset = 0;
        for (int i = 0; i < gpu; i++) {
            offset += distribution[i];
        }
        
        size_t chunkBytes = distribution[gpu] * sizeof(float);
        
        // Transfer data
        HIP_CHECK(hipMemcpyAsync(d_data[gpu], h_data + offset, chunkBytes,
                                hipMemcpyHostToDevice, manager.getStream(gpu)));
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((distribution[gpu] + block.x - 1) / block.x);
        hipLaunchKernelGGL(computeWorkload, grid, block, 0, manager.getStream(gpu),
                          d_data[gpu], distribution[gpu], 1000);
        
        // Transfer results back
        HIP_CHECK(hipMemcpyAsync(h_data + offset, d_data[gpu], chunkBytes,
                                hipMemcpyDeviceToHost, manager.getStream(gpu)));
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(manager.getStream(i)));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        if (distribution[i] > 0) {
            HIP_CHECK(hipFree(d_data[i]));
        }
    }
    
    double timeMs = duration.count() / 1000.0;
    printf("Multi-GPU weighted distribution time: %.2f ms\n\n", timeMs);
    
    return timeMs;
}

// Multi-GPU matrix multiplication
double runMultiGPUMatrixMul(MultiGPUManager& manager) {
    printf("Running multi-GPU matrix multiplication (%dx%d)...\n", MATRIX_SIZE, MATRIX_SIZE);
    
    int numGPUs = manager.getNumGPUs();
    size_t matrixBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // Allocate host matrices
    float *h_A = new float[MATRIX_SIZE * MATRIX_SIZE];
    float *h_B = new float[MATRIX_SIZE * MATRIX_SIZE];
    float *h_C = new float[MATRIX_SIZE * MATRIX_SIZE];
    
    // Initialize matrices
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
        h_C[i] = 0.0f;
    }
    
    // Divide matrix by rows across GPUs
    int rowsPerGPU = MATRIX_SIZE / numGPUs;
    std::vector<int> rowDistribution(numGPUs);
    
    for (int i = 0; i < numGPUs - 1; i++) {
        rowDistribution[i] = rowsPerGPU;
    }
    rowDistribution[numGPUs - 1] = MATRIX_SIZE - rowsPerGPU * (numGPUs - 1);
    
    printf("Matrix row distribution:\n");
    for (int i = 0; i < numGPUs; i++) {
        printf("  GPU %d: %d rows\n", i, rowDistribution[i]);
    }
    printf("\n");
    
    std::vector<float*> d_A(numGPUs), d_B(numGPUs), d_C(numGPUs);
    
    // Allocate device memory
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        
        size_t partialMatrixBytes = rowDistribution[i] * MATRIX_SIZE * sizeof(float);
        HIP_CHECK(hipMalloc(&d_A[i], partialMatrixBytes));
        HIP_CHECK(hipMalloc(&d_B[i], matrixBytes));  // Each GPU needs full B matrix
        HIP_CHECK(hipMalloc(&d_C[i], partialMatrixBytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parallel matrix multiplication
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        HIP_CHECK(hipSetDevice(gpu));
        
        int rowOffset = 0;
        for (int i = 0; i < gpu; i++) {
            rowOffset += rowDistribution[i];
        }
        
        size_t partialMatrixBytes = rowDistribution[gpu] * MATRIX_SIZE * sizeof(float);
        
        // Transfer data
        HIP_CHECK(hipMemcpyAsync(d_A[gpu], h_A + rowOffset * MATRIX_SIZE, 
                                partialMatrixBytes, hipMemcpyHostToDevice, manager.getStream(gpu)));
        HIP_CHECK(hipMemcpyAsync(d_B[gpu], h_B, matrixBytes, 
                                hipMemcpyHostToDevice, manager.getStream(gpu)));
        
        // Launch matrix multiplication kernel
        dim3 block(16, 16);
        dim3 grid((MATRIX_SIZE + block.x - 1) / block.x, 
                  (rowDistribution[gpu] + block.y - 1) / block.y);
        
        hipLaunchKernelGGL(matrixMul, grid, block, 0, manager.getStream(gpu),
                          d_A[gpu], d_B[gpu], d_C[gpu], MATRIX_SIZE, rowDistribution[gpu]);
        
        // Transfer result back
        HIP_CHECK(hipMemcpyAsync(h_C + rowOffset * MATRIX_SIZE, d_C[gpu], 
                                partialMatrixBytes, hipMemcpyDeviceToHost, manager.getStream(gpu)));
    }
    
    // Wait for completion
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(manager.getStream(i)));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate GFLOPS
    double gflops = (2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE) / (duration.count() / 1e6) / 1e9;
    
    printf("Multi-GPU matrix multiplication completed\n");
    printf("Time: %.2f ms\n", duration.count() / 1000.0);
    printf("Performance: %.1f GFLOPS\n\n", gflops);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(d_A[i]));
        HIP_CHECK(hipFree(d_B[i]));
        HIP_CHECK(hipFree(d_C[i]));
    }
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return duration.count() / 1000.0;
}

bool verifyResults(float *reference, float *result, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(reference[i] - result[i]) > tolerance) {
            printf("Verification failed at index %d: ref=%.6f, result=%.6f\n", 
                   i, reference[i], result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== HIP Multi-GPU Programming Demo ===\n\n");
    
    // Initialize multi-GPU manager
    MultiGPUManager manager;
    
    if (manager.getNumGPUs() < 1) {
        printf("Error: No GPUs detected!\n");
        return 1;
    }
    
    // Initialize data
    printf("Initializing vector data (%.1f MB)...\n", 
           VECTOR_SIZE * sizeof(float) / (1024.0f * 1024.0f));
    
    float *h_data_single = new float[VECTOR_SIZE];
    float *h_data_equal = new float[VECTOR_SIZE];
    float *h_data_weighted = new float[VECTOR_SIZE];
    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float value = sinf(i * 0.001f);
        h_data_single[i] = value;
        h_data_equal[i] = value;
        h_data_weighted[i] = value;
    }
    
    printf("\n=== Performance Comparison ===\n");
    
    // Single GPU baseline
    double singleTime = runSingleGPU(h_data_single, VECTOR_SIZE);
    
    if (manager.getNumGPUs() > 1) {
        // Multi-GPU equal distribution
        double equalTime = runMultiGPUEqual(manager, h_data_equal, VECTOR_SIZE);
        
        // Multi-GPU weighted distribution
        double weightedTime = runMultiGPUWeighted(manager, h_data_weighted, VECTOR_SIZE);
        
        // Verify results
        printf("=== Verification ===\n");
        bool equalValid = verifyResults(h_data_single, h_data_equal, VECTOR_SIZE);
        bool weightedValid = verifyResults(h_data_single, h_data_weighted, VECTOR_SIZE);
        
        printf("Equal distribution result: %s\n", equalValid ? "VALID" : "INVALID");
        printf("Weighted distribution result: %s\n", weightedValid ? "VALID" : "INVALID");
        printf("\n");
        
        // Performance analysis
        printf("=== Scaling Analysis ===\n");
        printf("Single GPU time: %.2f ms\n", singleTime);
        printf("Multi-GPU equal time: %.2f ms (%.1fx speedup, %.0f%% efficiency)\n", 
               equalTime, singleTime / equalTime, 
               100.0 * singleTime / equalTime / manager.getNumGPUs());
        printf("Multi-GPU weighted time: %.2f ms (%.1fx speedup, %.0f%% efficiency)\n", 
               weightedTime, singleTime / weightedTime,
               100.0 * singleTime / weightedTime / manager.getNumGPUs());
        printf("\n");
        
        // Matrix multiplication demo
        runMultiGPUMatrixMul(manager);
    } else {
        printf("Single GPU system - skipping multi-GPU tests\n\n");
    }
    
    printf("=== Key Insights ===\n");
    printf("- HIP enables efficient multi-GPU programming across AMD and NVIDIA hardware\n");
    printf("- Load balancing based on GPU capabilities improves performance\n");
    printf("- OpenMP provides easy CPU parallelization for GPU management\n");
    printf("- Asynchronous operations maximize multi-GPU utilization\n");
    
    if (manager.getNumGPUs() > 1) {
        printf("- Achieved multi-GPU scaling with %d GPUs\n", manager.getNumGPUs());
        printf("- Weighted distribution can optimize load balancing\n");
    }
    
    // Cleanup
    delete[] h_data_single;
    delete[] h_data_equal;
    delete[] h_data_weighted;
    
    printf("\n=== HIP Multi-GPU Demo Completed Successfully ===\n");
    return 0;
}