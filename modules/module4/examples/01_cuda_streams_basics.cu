#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define NUM_STREAMS 4
#define CHUNK_SIZE (1024 * 1024)  // 1M elements per chunk
#define TOTAL_SIZE (NUM_STREAMS * CHUNK_SIZE * 4)

// Simple kernel for demonstration
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate some work
        float temp = 0;
        for (int i = 0; i < 100; i++) {
            temp += sinf(a[idx] + b[idx] + i);
        }
        c[idx] = temp / 100.0f;
    }
}

// CPU implementation for verification
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        float temp = 0;
        for (int j = 0; j < 100; j++) {
            temp += sinf(a[i] + b[i] + j);
        }
        c[i] = temp / 100.0f;
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

// Synchronous version (baseline)
double runSynchronous(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running synchronous version...\n");
    
    float *d_a, *d_b, *d_c;
    size_t bytes = totalSize * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((totalSize + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, totalSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return time;
}

// Asynchronous version with streams
double runAsynchronous(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running asynchronous version with %d streams...\n", NUM_STREAMS);
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned host memory for faster transfers
    float *h_a_pinned, *h_b_pinned, *h_c_pinned;
    size_t bytes = totalSize * sizeof(float);
    
    CUDA_CHECK(cudaHostAlloc(&h_a_pinned, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b_pinned, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c_pinned, bytes, cudaHostAllocDefault));
    
    // Copy data to pinned memory
    memcpy(h_a_pinned, h_a, bytes);
    memcpy(h_b_pinned, h_b, bytes);
    
    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int chunkSize = totalSize / NUM_STREAMS;
    size_t chunkBytes = chunkSize * sizeof(float);
    
    // Launch operations on all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunkSize;
        
        // Async copy to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a_pinned + offset, 
                                  chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b_pinned + offset, 
                                  chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((chunkSize + block.x - 1) / block.x);
        vectorAdd<<<grid, block, 0, streams[i]>>>(d_a + offset, d_b + offset, 
                                                  d_c + offset, chunkSize);
        CUDA_CHECK(cudaGetLastError());
        
        // Async copy back
        CUDA_CHECK(cudaMemcpyAsync(h_c_pinned + offset, d_c + offset, 
                                  chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy result back to original host memory
    memcpy(h_c, h_c_pinned, bytes);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaFreeHost(h_a_pinned);
    cudaFreeHost(h_b_pinned);
    cudaFreeHost(h_c_pinned);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return time;
}

// Pipeline version - overlapping computation with data transfer
double runPipelined(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running pipelined version with overlapping...\n");
    
    const int numChunks = NUM_STREAMS * 3; // More chunks for better overlap
    const int chunkSize = totalSize / numChunks;
    const size_t chunkBytes = chunkSize * sizeof(float);
    
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned host memory
    float *h_a_pinned, *h_b_pinned, *h_c_pinned;
    size_t bytes = totalSize * sizeof(float);
    
    CUDA_CHECK(cudaHostAlloc(&h_a_pinned, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b_pinned, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c_pinned, bytes, cudaHostAllocDefault));
    
    memcpy(h_a_pinned, h_a, bytes);
    memcpy(h_b_pinned, h_b, bytes);
    
    // Allocate GPU memory (multiple buffers for pipelining)
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_b[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_c[i], chunkBytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pipeline execution
    for (int chunk = 0; chunk < numChunks; chunk++) {
        int streamId = chunk % NUM_STREAMS;
        int offset = chunk * chunkSize;
        
        // Stage 1: Transfer input data
        CUDA_CHECK(cudaMemcpyAsync(d_a[streamId], h_a_pinned + offset, 
                                  chunkBytes, cudaMemcpyHostToDevice, streams[streamId]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[streamId], h_b_pinned + offset, 
                                  chunkBytes, cudaMemcpyHostToDevice, streams[streamId]));
        
        // Stage 2: Process data
        dim3 block(256);
        dim3 grid((chunkSize + block.x - 1) / block.x);
        vectorAdd<<<grid, block, 0, streams[streamId]>>>(d_a[streamId], d_b[streamId], 
                                                         d_c[streamId], chunkSize);
        
        // Stage 3: Transfer result back
        CUDA_CHECK(cudaMemcpyAsync(h_c_pinned + offset, d_c[streamId], 
                                  chunkBytes, cudaMemcpyDeviceToHost, streams[streamId]));
    }
    
    // Wait for all streams
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    memcpy(h_c, h_c_pinned, bytes);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
    
    cudaFreeHost(h_a_pinned);
    cudaFreeHost(h_b_pinned);
    cudaFreeHost(h_c_pinned);
    
    return time;
}

// Stream priority demonstration
void demonstrateStreamPriorities() {
    printf("\nDemonstrating stream priorities...\n");
    
    // Get priority range
    int minPriority, maxPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
    
    printf("Stream priority range: %d (highest) to %d (lowest)\n", 
           maxPriority, minPriority);
    
    // Create streams with different priorities
    cudaStream_t highPriorityStream, lowPriorityStream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamDefault, maxPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamDefault, minPriority));
    
    // Launch kernels with different priorities
    dim3 block(256);
    dim3 grid(1024);
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * 256 * sizeof(float)));
    
    printf("Launching high priority kernel...\n");
    vectorAdd<<<grid, block, 0, highPriorityStream>>>(d_data, d_data, d_data, 1024 * 256);
    
    printf("Launching low priority kernel...\n");
    vectorAdd<<<grid, block, 0, lowPriorityStream>>>(d_data, d_data, d_data, 1024 * 256);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaStreamDestroy(highPriorityStream);
    cudaStreamDestroy(lowPriorityStream);
    cudaFree(d_data);
    
    printf("Priority demonstration completed.\n");
}

// Stream callback demonstration
void CUDART_CB streamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback executed! Status: %s, UserData: %s\n", 
           cudaGetErrorString(status), (char*)userData);
}

void demonstrateStreamCallbacks() {
    printf("\nDemonstrating stream callbacks...\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
    
    // Launch a simple kernel
    dim3 block(256);
    dim3 grid(4);
    vectorAdd<<<grid, block, 0, stream>>>(d_data, d_data, d_data, 1024);
    
    // Add callback
    char *message = (char*)"Kernel execution completed";
    CUDA_CHECK(cudaStreamAddCallback(stream, streamCallback, message, 0));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

bool verifyResults(float *result1, float *result2, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(result1[i] - result2[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, result1[i], result2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("CUDA Streams Basics Demonstration\n");
    printf("==================================\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async Engine Count: %d\n", prop.asyncEngineCount);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Memory Clock Rate: %.2f MHz\n\n", prop.memoryClockRate / 1000.0f);
    
    // Allocate host memory
    const int totalSize = TOTAL_SIZE;
    size_t bytes = totalSize * sizeof(float);
    
    printf("Processing %d elements (%.2f MB)\n", totalSize, bytes / (1024.0f * 1024.0f));
    printf("Using %d streams with chunks of %d elements each\n\n", NUM_STREAMS, totalSize / NUM_STREAMS);
    
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_sync = (float*)malloc(bytes);
    float *h_c_async = (float*)malloc(bytes);
    float *h_c_pipeline = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < totalSize; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }
    
    // Run different versions
    double syncTime = runSynchronous(h_a, h_b, h_c_sync, totalSize);
    double asyncTime = runAsynchronous(h_a, h_b, h_c_async, totalSize);
    double pipelineTime = runPipelined(h_a, h_b, h_c_pipeline, totalSize);
    
    printf("\nPerformance Results:\n");
    printf("Synchronous:  %.2f ms\n", syncTime);
    printf("Asynchronous: %.2f ms (%.2fx speedup)\n", asyncTime, syncTime / asyncTime);
    printf("Pipelined:    %.2f ms (%.2fx speedup)\n", pipelineTime, syncTime / pipelineTime);
    
    // Verify results
    printf("\nVerification:\n");
    bool syncAsyncMatch = verifyResults(h_c_sync, h_c_async, totalSize);
    bool syncPipelineMatch = verifyResults(h_c_sync, h_c_pipeline, totalSize);
    
    printf("Sync vs Async: %s\n", syncAsyncMatch ? "PASS" : "FAIL");
    printf("Sync vs Pipeline: %s\n", syncPipelineMatch ? "PASS" : "FAIL");
    
    // Calculate bandwidth
    double dataTransferred = 3.0 * bytes; // 2 inputs + 1 output
    printf("\nBandwidth Analysis:\n");
    printf("Data transferred: %.2f MB\n", dataTransferred / (1024.0 * 1024.0));
    printf("Synchronous bandwidth: %.2f GB/s\n", 
           (dataTransferred / (1024.0*1024.0*1024.0)) / (syncTime / 1000.0));
    printf("Asynchronous bandwidth: %.2f GB/s\n", 
           (dataTransferred / (1024.0*1024.0*1024.0)) / (asyncTime / 1000.0));
    printf("Pipelined bandwidth: %.2f GB/s\n", 
           (dataTransferred / (1024.0*1024.0*1024.0)) / (pipelineTime / 1000.0));
    
    // Demonstrate additional stream features
    demonstrateStreamPriorities();
    demonstrateStreamCallbacks();
    
    // Performance analysis
    printf("\nPerformance Analysis:\n");
    printf("- Asynchronous execution provides %.1fx speedup over synchronous\n", 
           syncTime / asyncTime);
    printf("- Pipelined execution provides %.1fx speedup over synchronous\n", 
           syncTime / pipelineTime);
    printf("- Overlap efficiency: %.1f%% improvement from pipelining\n", 
           ((syncTime - pipelineTime) / syncTime) * 100.0);
    
    printf("\nKey Insights:\n");
    printf("- Streams enable concurrent execution of kernels and memory transfers\n");
    printf("- Pinned memory is essential for async memory transfers\n");
    printf("- Pipeline processing maximizes GPU utilization\n");
    printf("- Stream priorities help manage resource contention\n");
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_sync);
    free(h_c_async);
    free(h_c_pipeline);
    
    printf("\nCUDA Streams demonstration completed successfully!\n");
    return 0;
}