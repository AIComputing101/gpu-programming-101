#include <hip/hip_runtime.h>
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

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Synchronous version (baseline)
double runSynchronous(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running synchronous version...\n");
    
    float *d_a, *d_b, *d_c;
    size_t bytes = totalSize * sizeof(float);
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((totalSize + block.x - 1) / block.x);
    hipLaunchKernelGGL(vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, totalSize);
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
    
    // Wait for completion
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    
    double timeMs = duration.count() / 1000.0;
    printf("Synchronous time: %.2f ms\n", timeMs);
    
    return timeMs;
}

// Asynchronous version with streams
double runAsynchronousStreams(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running asynchronous streams version...\n");
    
    // Create streams
    hipStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }
    
    // Allocate pinned host memory for faster transfers
    float *h_a_pinned, *h_b_pinned, *h_c_pinned;
    size_t bytes = totalSize * sizeof(float);
    
    HIP_CHECK(hipHostMalloc(&h_a_pinned, bytes));
    HIP_CHECK(hipHostMalloc(&h_b_pinned, bytes));
    HIP_CHECK(hipHostMalloc(&h_c_pinned, bytes));
    
    // Copy data to pinned memory
    memcpy(h_a_pinned, h_a, bytes);
    memcpy(h_b_pinned, h_b, bytes);
    
    // Allocate device memory for each stream
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    size_t chunkBytes = CHUNK_SIZE * sizeof(float);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipMalloc(&d_a[i], chunkBytes));
        HIP_CHECK(hipMalloc(&d_b[i], chunkBytes));
        HIP_CHECK(hipMalloc(&d_c[i], chunkBytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pipeline processing across multiple streams
    int numChunks = totalSize / CHUNK_SIZE;
    dim3 block(256);
    dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);
    
    for (int chunk = 0; chunk < numChunks; chunk++) {
        int streamId = chunk % NUM_STREAMS;
        int offset = chunk * CHUNK_SIZE;
        
        // Stage 1: Transfer input data
        HIP_CHECK(hipMemcpyAsync(d_a[streamId], h_a_pinned + offset, chunkBytes,
                                hipMemcpyHostToDevice, streams[streamId]));
        HIP_CHECK(hipMemcpyAsync(d_b[streamId], h_b_pinned + offset, chunkBytes,
                                hipMemcpyHostToDevice, streams[streamId]));
        
        // Stage 2: Process data
        hipLaunchKernelGGL(vectorAdd, grid, block, 0, streams[streamId],
                          d_a[streamId], d_b[streamId], d_c[streamId], CHUNK_SIZE);
        
        // Stage 3: Transfer results
        HIP_CHECK(hipMemcpyAsync(h_c_pinned + offset, d_c[streamId], chunkBytes,
                                hipMemcpyDeviceToHost, streams[streamId]));
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back to original host memory
    memcpy(h_c, h_c_pinned, bytes);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipFree(d_a[i]));
        HIP_CHECK(hipFree(d_b[i]));
        HIP_CHECK(hipFree(d_c[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
    }
    
    HIP_CHECK(hipHostFree(h_a_pinned));
    HIP_CHECK(hipHostFree(h_b_pinned));
    HIP_CHECK(hipHostFree(h_c_pinned));
    
    double timeMs = duration.count() / 1000.0;
    printf("Asynchronous streams time: %.2f ms\n", timeMs);
    
    return timeMs;
}

// Advanced asynchronous version with stream priorities
double runAsynchronousAdvanced(float *h_a, float *h_b, float *h_c, int totalSize) {
    printf("Running advanced asynchronous version with stream priorities...\n");
    
    // Check if stream priorities are supported
    int minPriority, maxPriority;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
    printf("Stream priority range: %d (highest) to %d (lowest)\n", minPriority, maxPriority);
    
    // Create streams with different priorities
    hipStream_t highPriorityStream, normalStreams[NUM_STREAMS - 1];
    
    HIP_CHECK(hipStreamCreateWithPriority(&highPriorityStream, hipStreamDefault, minPriority));
    for (int i = 0; i < NUM_STREAMS - 1; i++) {
        HIP_CHECK(hipStreamCreateWithPriority(&normalStreams[i], hipStreamDefault, maxPriority));
    }
    
    // Allocate pinned memory
    float *h_a_pinned, *h_b_pinned, *h_c_pinned;
    size_t bytes = totalSize * sizeof(float);
    
    HIP_CHECK(hipHostMalloc(&h_a_pinned, bytes));
    HIP_CHECK(hipHostMalloc(&h_b_pinned, bytes));
    HIP_CHECK(hipHostMalloc(&h_c_pinned, bytes));
    
    memcpy(h_a_pinned, h_a, bytes);
    memcpy(h_b_pinned, h_b, bytes);
    
    // Device memory allocation
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    size_t chunkBytes = CHUNK_SIZE * sizeof(float);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipMalloc(&d_a[i], chunkBytes));
        HIP_CHECK(hipMalloc(&d_b[i], chunkBytes));
        HIP_CHECK(hipMalloc(&d_c[i], chunkBytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process with stream priorities
    int numChunks = totalSize / CHUNK_SIZE;
    dim3 block(256);
    dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);
    
    for (int chunk = 0; chunk < numChunks; chunk++) {
        hipStream_t currentStream;
        int deviceId;
        
        if (chunk == 0) {
            // First chunk gets high priority
            currentStream = highPriorityStream;
            deviceId = 0;
        } else {
            currentStream = normalStreams[(chunk - 1) % (NUM_STREAMS - 1)];
            deviceId = (chunk - 1) % (NUM_STREAMS - 1) + 1;
        }
        
        int offset = chunk * CHUNK_SIZE;
        
        // Memory transfers and kernel launch
        HIP_CHECK(hipMemcpyAsync(d_a[deviceId], h_a_pinned + offset, chunkBytes,
                                hipMemcpyHostToDevice, currentStream));
        HIP_CHECK(hipMemcpyAsync(d_b[deviceId], h_b_pinned + offset, chunkBytes,
                                hipMemcpyHostToDevice, currentStream));
        
        hipLaunchKernelGGL(vectorAdd, grid, block, 0, currentStream,
                          d_a[deviceId], d_b[deviceId], d_c[deviceId], CHUNK_SIZE);
        
        HIP_CHECK(hipMemcpyAsync(h_c_pinned + offset, d_c[deviceId], chunkBytes,
                                hipMemcpyDeviceToHost, currentStream));
    }
    
    // Synchronize all streams
    HIP_CHECK(hipStreamSynchronize(highPriorityStream));
    for (int i = 0; i < NUM_STREAMS - 1; i++) {
        HIP_CHECK(hipStreamSynchronize(normalStreams[i]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back
    memcpy(h_c, h_c_pinned, bytes);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK(hipFree(d_a[i]));
        HIP_CHECK(hipFree(d_b[i]));
        HIP_CHECK(hipFree(d_c[i]));
    }
    
    HIP_CHECK(hipStreamDestroy(highPriorityStream));
    for (int i = 0; i < NUM_STREAMS - 1; i++) {
        HIP_CHECK(hipStreamDestroy(normalStreams[i]));
    }
    
    HIP_CHECK(hipHostFree(h_a_pinned));
    HIP_CHECK(hipHostFree(h_b_pinned));
    HIP_CHECK(hipHostFree(h_c_pinned));
    
    double timeMs = duration.count() / 1000.0;
    printf("Advanced asynchronous time: %.2f ms\n", timeMs);
    
    return timeMs;
}

// Verification function
bool verifyResults(float *gpu_result, float *cpu_result, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Verification failed at index %d: GPU=%.6f, CPU=%.6f\n", 
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

void printDeviceInfo() {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    printf("=== ROCm/HIP Device Information ===\n");
    printf("Number of devices: %d\n", deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));
        
        printf("\nDevice %d: %s\n", device, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.1f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        
        // Check for concurrent kernels support
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
    }
    printf("\n");
}

int main() {
    printf("=== HIP Streams and Asynchronous Execution Demo ===\n\n");
    
    // Print device information
    printDeviceInfo();
    
    // Initialize data
    printf("Initializing data (%.1f MB total)...\n", 
           TOTAL_SIZE * sizeof(float) / (1024.0f * 1024.0f));
    
    float *h_a = new float[TOTAL_SIZE];
    float *h_b = new float[TOTAL_SIZE];
    float *h_c_sync = new float[TOTAL_SIZE];
    float *h_c_async = new float[TOTAL_SIZE];
    float *h_c_advanced = new float[TOTAL_SIZE];
    float *h_c_cpu = new float[TOTAL_SIZE];
    
    // Initialize input data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }
    
    printf("Calculating reference result on CPU...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Calculate subset for verification (full CPU calculation would take too long)
    int verifySize = std::min(TOTAL_SIZE, 10000);
    vectorAddCPU(h_a, h_b, h_c_cpu, verifySize);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    printf("CPU reference time (first %d elements): %.2f ms\n\n", 
           verifySize, cpu_duration.count() / 1000.0);
    
    // Run performance comparison
    printf("=== Performance Comparison ===\n");
    double syncTime = runSynchronous(h_a, h_b, h_c_sync, TOTAL_SIZE);
    printf("\n");
    
    double asyncTime = runAsynchronousStreams(h_a, h_b, h_c_async, TOTAL_SIZE);
    printf("\n");
    
    double advancedTime = runAsynchronousAdvanced(h_a, h_b, h_c_advanced, TOTAL_SIZE);
    printf("\n");
    
    // Verify results
    printf("=== Verification ===\n");
    bool syncValid = verifyResults(h_c_sync, h_c_cpu, verifySize);
    bool asyncValid = verifyResults(h_c_async, h_c_cpu, verifySize);
    bool advancedValid = verifyResults(h_c_advanced, h_c_cpu, verifySize);
    
    printf("Synchronous result: %s\n", syncValid ? "VALID" : "INVALID");
    printf("Asynchronous streams result: %s\n", asyncValid ? "VALID" : "INVALID");
    printf("Advanced async result: %s\n", advancedValid ? "VALID" : "INVALID");
    printf("\n");
    
    // Performance analysis
    printf("=== Performance Analysis ===\n");
    printf("Synchronous time: %.2f ms\n", syncTime);
    printf("Async streams time: %.2f ms (%.1fx speedup)\n", 
           asyncTime, syncTime / asyncTime);
    printf("Advanced async time: %.2f ms (%.1fx speedup)\n", 
           advancedTime, syncTime / advancedTime);
    
    double bandwidth = (TOTAL_SIZE * 3 * sizeof(float)) / (asyncTime / 1000.0) / (1024*1024*1024);
    printf("Effective bandwidth (async streams): %.1f GB/s\n", bandwidth);
    
    printf("\nKey Insights:\n");
    printf("- HIP streams enable overlapping of memory transfers and computation\n");
    printf("- Pinned memory improves transfer performance significantly\n");
    printf("- Stream priorities can help optimize critical path execution\n");
    printf("- Pipeline processing maximizes GPU utilization\n");
    
    if (asyncTime < syncTime) {
        printf("- Achieved %.1fx speedup through asynchronous execution\n", syncTime / asyncTime);
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_sync;
    delete[] h_c_async;
    delete[] h_c_advanced;
    delete[] h_c_cpu;
    
    printf("\n=== HIP Streams Demo Completed Successfully ===\n");
    return 0;
}