#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

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
    extern __shared__ float shared[];
    
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
__global__ void computeWorkload(float *data, int n, int workAmount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < workAmount; i++) {
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

void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Multi-GPU System Information:\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        
        // Check P2P capabilities with other devices
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                printf("  P2P Access to Device %d: %s\n", j, canAccessPeer ? "Yes" : "No");
            }
        }
        printf("\n");
    }
}

// Single GPU baseline
double runSingleGPU(float *h_data, int size) {
    printf("Running single GPU baseline...\n");
    
    float *d_data;
    size_t bytes = size * sizeof(float);
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    computeWorkload<<<grid, block>>>(d_data, size, 1000);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    cudaFree(d_data);
    return time;
}

// Multi-GPU with equal distribution
double runMultiGPUEqual(float *h_data, int size, int numGPUs) {
    printf("Running multi-GPU with equal distribution (%d GPUs)...\n", numGPUs);
    
    if (numGPUs <= 1) return runSingleGPU(h_data, size);
    
    int chunkSize = size / numGPUs;
    size_t chunkBytes = chunkSize * sizeof(float);
    
    float **d_data = new float*[numGPUs];
    cudaStream_t *streams = new cudaStream_t[numGPUs];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch on all GPUs in parallel
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        CUDA_CHECK(cudaMalloc(&d_data[gpu], chunkBytes));
        
        int offset = gpu * chunkSize;
        
        // Copy data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_data[gpu], h_data + offset, 
                                  chunkBytes, cudaMemcpyHostToDevice, streams[gpu]));
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((chunkSize + block.x - 1) / block.x);
        computeWorkload<<<grid, block, 0, streams[gpu]>>>(d_data[gpu], chunkSize, 1000);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data[gpu], 
                                  chunkBytes, cudaMemcpyDeviceToHost, streams[gpu]));
    }
    
    // Wait for all GPUs to complete
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(d_data[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    delete[] d_data;
    delete[] streams;
    
    return time;
}

// Multi-GPU with weighted distribution based on compute capability
double runMultiGPUWeighted(float *h_data, int size, int numGPUs) {
    printf("Running multi-GPU with weighted distribution...\n");
    
    if (numGPUs <= 1) return runSingleGPU(h_data, size);
    
    // Calculate weights based on compute capability
    double *weights = new double[numGPUs];
    double totalWeight = 0.0;
    
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
        
        // Simple weight based on SM count and clock rate
        weights[gpu] = prop.multiProcessorCount * (prop.clockRate / 1000.0);
        totalWeight += weights[gpu];
    }
    
    // Normalize weights and calculate chunk sizes
    int *chunkSizes = new int[numGPUs];
    int totalAssigned = 0;
    
    for (int gpu = 0; gpu < numGPUs - 1; gpu++) {
        chunkSizes[gpu] = (int)((weights[gpu] / totalWeight) * size);
        totalAssigned += chunkSizes[gpu];
    }
    chunkSizes[numGPUs - 1] = size - totalAssigned; // Remainder goes to last GPU
    
    printf("Weighted distribution:\n");
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        printf("  GPU %d: %d elements (%.1f%%)\n", gpu, chunkSizes[gpu], 
               (chunkSizes[gpu] * 100.0) / size);
    }
    
    float **d_data = new float*[numGPUs];
    cudaStream_t *streams = new cudaStream_t[numGPUs];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch on all GPUs with weighted distribution
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        int currentOffset = 0;
        for (int i = 0; i < gpu; i++) {
            currentOffset += chunkSizes[i];
        }
        
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        CUDA_CHECK(cudaMalloc(&d_data[gpu], chunkSizes[gpu] * sizeof(float)));
        
        // Copy data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_data[gpu], h_data + currentOffset, 
                                  chunkSizes[gpu] * sizeof(float), 
                                  cudaMemcpyHostToDevice, streams[gpu]));
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((chunkSizes[gpu] + block.x - 1) / block.x);
        computeWorkload<<<grid, block, 0, streams[gpu]>>>(d_data[gpu], chunkSizes[gpu], 1000);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpyAsync(h_data + currentOffset, d_data[gpu], 
                                  chunkSizes[gpu] * sizeof(float), 
                                  cudaMemcpyDeviceToHost, streams[gpu]));
    }
    
    // Wait for all GPUs
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(d_data[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    delete[] d_data;
    delete[] streams;
    delete[] weights;
    delete[] chunkSizes;
    
    return time;
}

// Multi-GPU matrix multiplication
double runMultiGPUMatrixMul(float *h_A, float *h_B, float *h_C, int size, int numGPUs) {
    printf("Running multi-GPU matrix multiplication (%dx%d)...\n", size, size);
    
    if (numGPUs <= 1) {
        // Single GPU fallback
        float *d_A, *d_B, *d_C;
        size_t bytes = size * size * sizeof(float);
        
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
        matrixMul<<<grid, block>>>(d_A, d_B, d_C, size, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        return time;
    }
    
    // Multi-GPU implementation - row-wise distribution
    int rowsPerGPU = size / numGPUs;
    size_t matrixBytes = size * size * sizeof(float);
    size_t chunkABytes = rowsPerGPU * size * sizeof(float);
    size_t chunkCBytes = rowsPerGPU * size * sizeof(float);
    
    float **d_A = new float*[numGPUs];
    float **d_B = new float*[numGPUs];
    float **d_C = new float*[numGPUs];
    cudaStream_t *streams = new cudaStream_t[numGPUs];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        
        CUDA_CHECK(cudaMalloc(&d_A[gpu], chunkABytes));
        CUDA_CHECK(cudaMalloc(&d_B[gpu], matrixBytes)); // Full matrix B needed
        CUDA_CHECK(cudaMalloc(&d_C[gpu], chunkCBytes));
        
        int rowOffset = gpu * rowsPerGPU;
        
        // Copy chunk of A and full B matrix
        CUDA_CHECK(cudaMemcpyAsync(d_A[gpu], h_A + rowOffset * size, 
                                  chunkABytes, cudaMemcpyHostToDevice, streams[gpu]));
        CUDA_CHECK(cudaMemcpyAsync(d_B[gpu], h_B, 
                                  matrixBytes, cudaMemcpyHostToDevice, streams[gpu]));
        
        // Launch matrix multiplication
        dim3 block(16, 16);
        dim3 grid((size + block.x - 1) / block.x, (rowsPerGPU + block.y - 1) / block.y);
        matrixMul<<<grid, block, 0, streams[gpu]>>>(d_A[gpu], d_B[gpu], d_C[gpu], 
                                                    size, rowsPerGPU);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpyAsync(h_C + rowOffset * size, d_C[gpu], 
                                  chunkCBytes, cudaMemcpyDeviceToHost, streams[gpu]));
    }
    
    // Wait for all GPUs
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(d_A[gpu]);
        cudaFree(d_B[gpu]);
        cudaFree(d_C[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
    delete[] streams;
    
    return time;
}

// Multi-GPU reduction with aggregation
double runMultiGPUReduction(float *h_data, int size, int numGPUs, float *result) {
    printf("Running multi-GPU reduction...\n");
    
    if (numGPUs <= 1) {
        // Single GPU implementation
        float *d_input, *d_output;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, numBlocks * sizeof(float)));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaMemcpy(d_input, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
        
        vectorReduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, size);
        CUDA_CHECK(cudaGetLastError());
        
        // Final reduction on CPU
        float *h_partial = (float*)malloc(numBlocks * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_partial, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        *result = 0.0f;
        for (int i = 0; i < numBlocks; i++) {
            *result += h_partial[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        
        free(h_partial);
        cudaFree(d_input);
        cudaFree(d_output);
        
        return time;
    }
    
    // Multi-GPU reduction
    int chunkSize = size / numGPUs;
    int blockSize = 256;
    
    float **d_input = new float*[numGPUs];
    float **d_output = new float*[numGPUs];
    float *partial_results = new float[numGPUs];
    cudaStream_t *streams = new cudaStream_t[numGPUs];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        
        int currentChunkSize = (gpu == numGPUs - 1) ? size - gpu * chunkSize : chunkSize;
        int numBlocks = (currentChunkSize + blockSize - 1) / blockSize;
        
        CUDA_CHECK(cudaMalloc(&d_input[gpu], currentChunkSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output[gpu], numBlocks * sizeof(float)));
        
        // Copy data chunk
        CUDA_CHECK(cudaMemcpyAsync(d_input[gpu], h_data + gpu * chunkSize, 
                                  currentChunkSize * sizeof(float), 
                                  cudaMemcpyHostToDevice, streams[gpu]));
        
        // Launch reduction kernel
        vectorReduction<<<numBlocks, blockSize, blockSize * sizeof(float), streams[gpu]>>>(
            d_input[gpu], d_output[gpu], currentChunkSize);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy partial results back
        float *h_partial = (float*)malloc(numBlocks * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(h_partial, d_output[gpu], 
                                  numBlocks * sizeof(float), 
                                  cudaMemcpyDeviceToHost, streams[gpu]));
        
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
        
        // Final reduction on CPU for this GPU
        partial_results[gpu] = 0.0f;
        for (int i = 0; i < numBlocks; i++) {
            partial_results[gpu] += h_partial[i];
        }
        
        free(h_partial);
    }
    
    // Final aggregation
    *result = 0.0f;
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        *result += partial_results[gpu];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Cleanup
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(d_input[gpu]);
        cudaFree(d_output[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    delete[] d_input;
    delete[] d_output;
    delete[] partial_results;
    delete[] streams;
    
    return time;
}

void testScaling(float *h_data, int size) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("\nScaling Analysis:\n");
    printf("GPUs | Equal Dist. | Weighted Dist. | Speedup | Efficiency\n");
    printf("-----|-------------|----------------|---------|----------\n");
    
    double baseTime = runSingleGPU(h_data, size);
    
    for (int numGPUs = 1; numGPUs <= deviceCount; numGPUs++) {
        // Reset data for consistent testing
        for (int i = 0; i < size; i++) {
            h_data[i] = sinf(i * 0.001f);
        }
        
        double equalTime = runMultiGPUEqual(h_data, size, numGPUs);
        
        // Reset data again
        for (int i = 0; i < size; i++) {
            h_data[i] = sinf(i * 0.001f);
        }
        
        double weightedTime = runMultiGPUWeighted(h_data, size, numGPUs);
        
        double speedupEqual = baseTime / equalTime;
        double speedupWeighted = baseTime / weightedTime;
        double efficiency = speedupEqual / numGPUs * 100.0;
        
        printf(" %2d  | %8.2f ms | %9.2f ms | %6.2fx | %7.1f%%\n", 
               numGPUs, equalTime, weightedTime, speedupEqual, efficiency);
    }
}

int main() {
    printf("Multi-GPU Programming Demonstration\n");
    printf("===================================\n\n");
    
    printDeviceInfo();
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount < 2) {
        printf("This demo requires at least 2 GPUs. Found %d GPU(s).\n", deviceCount);
        printf("Running in single GPU mode for demonstration...\n\n");
        deviceCount = 1;
    }
    
    // Test 1: Vector processing with different load balancing strategies
    printf("=== Test 1: Vector Processing Load Balancing ===\n");
    
    const int vectorSize = VECTOR_SIZE;
    float *h_vector = (float*)malloc(vectorSize * sizeof(float));
    
    // Initialize vector
    for (int i = 0; i < vectorSize; i++) {
        h_vector[i] = sinf(i * 0.001f);
    }
    
    printf("Vector size: %d elements (%.2f MB)\n\n", 
           vectorSize, (vectorSize * sizeof(float)) / (1024.0f * 1024.0f));
    
    testScaling(h_vector, vectorSize);
    
    // Test 2: Matrix multiplication
    if (deviceCount >= 2) {
        printf("\n=== Test 2: Multi-GPU Matrix Multiplication ===\n");
        
        const int matrixSize = MATRIX_SIZE;
        const int matrixElements = matrixSize * matrixSize;
        const size_t matrixBytes = matrixElements * sizeof(float);
        
        printf("Matrix size: %dx%d (%.2f MB per matrix)\n", 
               matrixSize, matrixSize, matrixBytes / (1024.0f * 1024.0f));
        
        float *h_A = (float*)malloc(matrixBytes);
        float *h_B = (float*)malloc(matrixBytes);
        float *h_C = (float*)malloc(matrixBytes);
        
        // Initialize matrices
        for (int i = 0; i < matrixElements; i++) {
            h_A[i] = (float)(rand() % 100) / 100.0f;
            h_B[i] = (float)(rand() % 100) / 100.0f;
        }
        
        printf("\nMatrix Multiplication Results:\n");
        printf("GPUs | Time (ms) | Speedup | GFLOPS\n");
        printf("-----|-----------|---------|-------\n");
        
        for (int numGPUs = 1; numGPUs <= deviceCount; numGPUs++) {
            double time = runMultiGPUMatrixMul(h_A, h_B, h_C, matrixSize, numGPUs);
            double operations = 2.0 * matrixSize * matrixSize * matrixSize; // 2n³ operations
            double gflops = (operations / 1e9) / (time / 1000.0);
            double speedup = (numGPUs == 1) ? 1.0 : 
                           (runMultiGPUMatrixMul(h_A, h_B, h_C, matrixSize, 1) / time);
            
            printf(" %2d  | %8.2f  | %6.2fx | %6.1f\n", 
                   numGPUs, time, speedup, gflops);
        }
        
        free(h_A);
        free(h_B);
        free(h_C);
    }
    
    // Test 3: Multi-GPU reduction
    if (deviceCount >= 2) {
        printf("\n=== Test 3: Multi-GPU Reduction ===\n");
        
        const int reductionSize = VECTOR_SIZE;
        float *h_reduction_data = (float*)malloc(reductionSize * sizeof(float));
        
        // Initialize with known values for verification
        for (int i = 0; i < reductionSize; i++) {
            h_reduction_data[i] = 1.0f; // Sum should equal reductionSize
        }
        
        printf("Reduction size: %d elements\n", reductionSize);
        printf("Expected result: %.0f\n\n", (float)reductionSize);
        
        printf("GPUs | Time (ms) | Result    | Speedup | Accuracy\n");
        printf("-----|-----------|-----------|---------|----------\n");
        
        double baseTime = 0.0;
        for (int numGPUs = 1; numGPUs <= deviceCount; numGPUs++) {
            float result;
            double time = runMultiGPUReduction(h_reduction_data, reductionSize, numGPUs, &result);
            
            if (numGPUs == 1) baseTime = time;
            double speedup = baseTime / time;
            double accuracy = (result / reductionSize) * 100.0;
            
            printf(" %2d  | %8.2f  | %8.0f  | %6.2fx | %7.2f%%\n", 
                   numGPUs, time, result, speedup, accuracy);
        }
        
        free(h_reduction_data);
    }
    
    // Performance analysis
    printf("\n=== Performance Analysis ===\n");
    printf("Key Insights:\n");
    printf("1. Load balancing significantly affects multi-GPU performance\n");
    printf("2. Weighted distribution can improve performance on heterogeneous systems\n");
    printf("3. Communication overhead limits scaling efficiency\n");
    printf("4. Memory bandwidth often becomes the bottleneck\n");
    printf("5. Problem size must be large enough to overcome overhead\n");
    
    printf("\nOptimization Strategies:\n");
    printf("- Use asynchronous operations to hide latency\n");
    printf("- Minimize data transfers between GPUs\n");
    printf("- Balance computation based on GPU capabilities\n");
    printf("- Consider NUMA topology in multi-GPU systems\n");
    printf("- Use pinned memory for faster host-device transfers\n");
    
    free(h_vector);
    
    printf("\nMulti-GPU programming demonstration completed!\n");
    return 0;
}