#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define TRANSFER_SIZE (64 * 1024 * 1024)  // 64 MB
#define NUM_ITERATIONS 10

// Simple kernel for data processing
__global__ void processData(float *data, int n, int gpu_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple computation with GPU identifier
        data[idx] = data[idx] * (gpu_id + 1.0f) + sinf(idx * 0.001f);
    }
}

// Kernel to verify data integrity
__global__ void verifyData(float *data, float *expected, bool *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = fabsf(data[idx] - expected[idx]);
        if (diff > 1e-5f) {
            *result = false;
        }
    }
}

// Simple addition kernel for peer-to-peer communication
__global__ void addArrays(float *result, float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] += input[idx];
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

void checkP2PCapabilities() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Peer-to-Peer Capabilities Analysis:\n");
    printf("===================================\n");
    printf("Number of GPUs: %d\n\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("P2P communication requires at least 2 GPUs.\n");
        return;
    }
    
    // Check P2P access capabilities
    printf("P2P Access Matrix:\n");
    printf("From\\To  ");
    for (int j = 0; j < deviceCount; j++) {
        printf("GPU%d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < deviceCount; i++) {
        printf("GPU%d     ", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) {
                printf("---- ");
            } else {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                printf(" %s  ", canAccessPeer ? "Yes" : "No");
            }
        }
        printf("\n");
    }
    
    // Check detailed P2P attributes
    printf("\nDetailed P2P Attributes:\n");
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                printf("GPU %d to GPU %d:\n", i, j);
                
                int attr;
                CUDA_CHECK(cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrPerformanceRank, i, j));
                printf("  Performance Rank: %d\n", attr);
                
                CUDA_CHECK(cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrAccessSupported, i, j));
                printf("  Access Supported: %s\n", attr ? "Yes" : "No");
                
                CUDA_CHECK(cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrNativeAtomicSupported, i, j));
                printf("  Native Atomics: %s\n", attr ? "Yes" : "No");
                
                CUDA_CHECK(cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrCudaArrayAccessSupported, i, j));
                printf("  CUDA Array Access: %s\n", attr ? "Yes" : "No");
                
                printf("\n");
            }
        }
    }
}

void enableP2PAccess(int deviceCount) {
    printf("Enabling P2P access between all capable GPU pairs...\n");
    
    for (int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                
                if (canAccessPeer) {
                    cudaError_t result = cudaDeviceEnablePeerAccess(j, 0);
                    if (result == cudaSuccess) {
                        printf("  GPU %d -> GPU %d: P2P enabled\n", i, j);
                    } else if (result == cudaErrorPeerAccessAlreadyEnabled) {
                        printf("  GPU %d -> GPU %d: P2P already enabled\n", i, j);
                    } else {
                        printf("  GPU %d -> GPU %d: P2P enable failed (%s)\n", 
                               i, j, cudaGetErrorString(result));
                    }
                }
            }
        }
    }
    printf("\n");
}

// Measure P2P bandwidth between two GPUs
double measureP2PBandwidth(int srcDevice, int dstDevice, size_t bytes) {
    CUDA_CHECK(cudaSetDevice(srcDevice));
    
    float *src_data;
    CUDA_CHECK(cudaMalloc(&src_data, bytes));
    
    CUDA_CHECK(cudaSetDevice(dstDevice));
    
    float *dst_data;
    CUDA_CHECK(cudaMalloc(&dst_data, bytes));
    
    // Initialize source data
    CUDA_CHECK(cudaSetDevice(srcDevice));
    CUDA_CHECK(cudaMemset(src_data, 1, bytes));
    
    // Warm up
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpyPeer(dst_data, dstDevice, src_data, srcDevice, bytes));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpyPeer(dst_data, dstDevice, src_data, srcDevice, bytes));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double>(end - start).count();
    double bandwidth = (bytes * NUM_ITERATIONS) / (total_time * 1e9); // GB/s
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(srcDevice));
    cudaFree(src_data);
    CUDA_CHECK(cudaSetDevice(dstDevice));
    cudaFree(dst_data);
    
    return bandwidth;
}

// Measure host-to-device bandwidth for comparison
double measureHostDeviceBandwidth(int device, size_t bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    
    // Allocate host memory (pinned for better performance)
    float *host_data;
    CUDA_CHECK(cudaHostAlloc(&host_data, bytes, cudaHostAllocDefault));
    
    float *device_data;
    CUDA_CHECK(cudaMalloc(&device_data, bytes));
    
    // Initialize host data
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        host_data[i] = (float)i;
    }
    
    // Warm up
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration<double>(end - start).count();
    double bandwidth = (bytes * NUM_ITERATIONS) / (total_time * 1e9); // GB/s
    
    // Cleanup
    cudaFreeHost(host_data);
    cudaFree(device_data);
    
    return bandwidth;
}

// Demonstrate asynchronous P2P transfers
void demonstrateAsyncP2P(int srcDevice, int dstDevice) {
    printf("Demonstrating asynchronous P2P transfers (GPU %d -> GPU %d)...\n", 
           srcDevice, dstDevice);
    
    const size_t chunkSize = TRANSFER_SIZE / 4;
    const int numChunks = 4;
    
    // Allocate memory on both devices
    CUDA_CHECK(cudaSetDevice(srcDevice));
    float *src_data;
    CUDA_CHECK(cudaMalloc(&src_data, TRANSFER_SIZE));
    
    CUDA_CHECK(cudaSetDevice(dstDevice));
    float *dst_data;
    CUDA_CHECK(cudaMalloc(&dst_data, TRANSFER_SIZE));
    
    // Create streams
    cudaStream_t srcStream, dstStream;
    CUDA_CHECK(cudaSetDevice(srcDevice));
    CUDA_CHECK(cudaStreamCreate(&srcStream));
    CUDA_CHECK(cudaSetDevice(dstDevice));
    CUDA_CHECK(cudaStreamCreate(&dstStream));
    
    // Initialize source data
    CUDA_CHECK(cudaSetDevice(srcDevice));
    dim3 block(256);
    dim3 grid((TRANSFER_SIZE / sizeof(float) + block.x - 1) / block.x);
    processData<<<grid, block>>>(src_data, TRANSFER_SIZE / sizeof(float), srcDevice);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pipeline P2P transfers with computation
    for (int chunk = 0; chunk < numChunks; chunk++) {
        size_t offset = chunk * chunkSize;
        
        // Async P2P copy
        CUDA_CHECK(cudaMemcpyPeerAsync(dst_data + offset / sizeof(float), dstDevice,
                                      src_data + offset / sizeof(float), srcDevice,
                                      chunkSize, srcStream));
        
        // Process data on destination device
        CUDA_CHECK(cudaSetDevice(dstDevice));
        dim3 chunkGrid((chunkSize / sizeof(float) + block.x - 1) / block.x);
        processData<<<chunkGrid, block, 0, dstStream>>>(
            dst_data + offset / sizeof(float), 
            chunkSize / sizeof(float), 
            dstDevice);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Wait for all operations to complete
    CUDA_CHECK(cudaSetDevice(srcDevice));
    CUDA_CHECK(cudaStreamSynchronize(srcStream));
    CUDA_CHECK(cudaSetDevice(dstDevice));
    CUDA_CHECK(cudaStreamSynchronize(dstStream));
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("  Async P2P pipeline completed in %.2f ms\n", time);
    printf("  Effective bandwidth: %.2f GB/s\n", 
           (TRANSFER_SIZE / (1024.0*1024.0*1024.0)) / (time / 1000.0));
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(srcDevice));
    cudaFree(src_data);
    cudaStreamDestroy(srcStream);
    
    CUDA_CHECK(cudaSetDevice(dstDevice));
    cudaFree(dst_data);
    cudaStreamDestroy(dstStream);
}

// Multi-GPU ring communication pattern
void demonstrateRingCommunication(int deviceCount) {
    if (deviceCount < 3) {
        printf("Ring communication requires at least 3 GPUs.\n");
        return;
    }
    
    printf("Demonstrating ring communication pattern with %d GPUs...\n", deviceCount);
    
    const size_t elementsPerGPU = TRANSFER_SIZE / (deviceCount * sizeof(float));
    const size_t bytesPerGPU = elementsPerGPU * sizeof(float);
    
    // Allocate data on each GPU
    float **gpu_data = new float*[deviceCount];
    cudaStream_t *streams = new cudaStream_t[deviceCount];
    
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMalloc(&gpu_data[gpu], bytesPerGPU));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        
        // Initialize data with GPU-specific pattern
        dim3 block(256);
        dim3 grid((elementsPerGPU + block.x - 1) / block.x);
        processData<<<grid, block>>>(gpu_data[gpu], elementsPerGPU, gpu);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    printf("Initial data distribution:\n");
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        printf("  GPU %d: %.2f MB\n", gpu, bytesPerGPU / (1024.0*1024.0));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform ring communication: each GPU sends to next GPU in ring
    for (int step = 0; step < deviceCount - 1; step++) {
        printf("Ring step %d: ", step + 1);
        
        #pragma omp parallel for
        for (int gpu = 0; gpu < deviceCount; gpu++) {
            int nextGPU = (gpu + 1) % deviceCount;
            
            CUDA_CHECK(cudaSetDevice(gpu));
            
            // Allocate temporary buffer for received data
            float *temp_buffer;
            CUDA_CHECK(cudaMalloc(&temp_buffer, bytesPerGPU));
            
            // Send data to next GPU in ring
            CUDA_CHECK(cudaMemcpyPeerAsync(temp_buffer, nextGPU,
                                          gpu_data[gpu], gpu,
                                          bytesPerGPU, streams[gpu]));
            
            CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
            
            // Copy received data to main buffer on destination GPU
            CUDA_CHECK(cudaSetDevice(nextGPU));
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[nextGPU], temp_buffer,
                                      bytesPerGPU, cudaMemcpyDeviceToDevice, 
                                      streams[nextGPU]));
            
            CUDA_CHECK(cudaSetDevice(gpu));
            cudaFree(temp_buffer);
        }
        
        // Synchronize all GPUs
        for (int gpu = 0; gpu < deviceCount; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
        }
        
        printf("completed\n");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Ring communication completed in %.2f ms\n", time);
    printf("Total data transferred: %.2f MB\n", 
           (bytesPerGPU * deviceCount * (deviceCount - 1)) / (1024.0*1024.0));
    printf("Average bandwidth: %.2f GB/s\n", 
           (bytesPerGPU * deviceCount * (deviceCount - 1)) / 
           (1024.0*1024.0*1024.0) / (time / 1000.0));
    
    // Cleanup
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(gpu_data[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    delete[] gpu_data;
    delete[] streams;
}

// All-reduce operation using P2P
void demonstrateAllReduce(int deviceCount) {
    if (deviceCount < 2) {
        printf("All-reduce requires at least 2 GPUs.\n");
        return;
    }
    
    printf("Demonstrating all-reduce operation with %d GPUs...\n", deviceCount);
    
    const int elementsPerGPU = 1024 * 1024; // 1M floats per GPU
    const size_t bytesPerGPU = elementsPerGPU * sizeof(float);
    
    float **gpu_data = new float*[deviceCount];
    float **gpu_result = new float*[deviceCount];
    float *host_verify = new float[elementsPerGPU];
    
    // Initialize data on each GPU
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMalloc(&gpu_data[gpu], bytesPerGPU));
        CUDA_CHECK(cudaMalloc(&gpu_result[gpu], bytesPerGPU));
        
        // Initialize with GPU-specific values (gpu + 1)
        float *temp_data = new float[elementsPerGPU];
        for (int i = 0; i < elementsPerGPU; i++) {
            temp_data[i] = (float)(gpu + 1);
        }
        
        CUDA_CHECK(cudaMemcpy(gpu_data[gpu], temp_data, bytesPerGPU, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_result[gpu], temp_data, bytesPerGPU, cudaMemcpyHostToDevice));
        
        delete[] temp_data;
    }
    
    // Calculate expected result for verification
    float expected_sum = 0.0f;
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        expected_sum += (gpu + 1);
    }
    
    for (int i = 0; i < elementsPerGPU; i++) {
        host_verify[i] = expected_sum;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform all-reduce: sum data from all GPUs
    for (int srcGPU = 0; srcGPU < deviceCount; srcGPU++) {
        for (int dstGPU = 0; dstGPU < deviceCount; dstGPU++) {
            if (srcGPU != dstGPU) {
                // Allocate temporary buffer on destination GPU
                CUDA_CHECK(cudaSetDevice(dstGPU));
                float *temp_buffer;
                CUDA_CHECK(cudaMalloc(&temp_buffer, bytesPerGPU));
                
                // Copy data from source to destination
                CUDA_CHECK(cudaMemcpyPeer(temp_buffer, dstGPU,
                                         gpu_data[srcGPU], srcGPU, bytesPerGPU));
                
                // Add to result (element-wise addition)
                dim3 block(256);
                dim3 grid((elementsPerGPU + block.x - 1) / block.x);
                
                // Launch addition kernel
                addArrays<<<grid, block>>>(gpu_result[dstGPU], temp_buffer, elementsPerGPU);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                
                cudaFree(temp_buffer);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Verify results
    printf("All-reduce completed in %.2f ms\n", time);
    
    bool all_correct = true;
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        float *host_result = new float[elementsPerGPU];
        CUDA_CHECK(cudaMemcpy(host_result, gpu_result[gpu], bytesPerGPU, 
                             cudaMemcpyDeviceToHost));
        
        bool gpu_correct = true;
        for (int i = 0; i < 100; i++) { // Check first 100 elements
            if (fabsf(host_result[i] - expected_sum) > 1e-5f) {
                gpu_correct = false;
                break;
            }
        }
        
        printf("  GPU %d result verification: %s (expected %.1f, got %.1f)\n", 
               gpu, gpu_correct ? "PASS" : "FAIL", expected_sum, host_result[0]);
        
        if (!gpu_correct) all_correct = false;
        delete[] host_result;
    }
    
    printf("Overall verification: %s\n", all_correct ? "PASS" : "FAIL");
    
    // Cleanup
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        cudaFree(gpu_data[gpu]);
        cudaFree(gpu_result[gpu]);
    }
    
    delete[] gpu_data;
    delete[] gpu_result;
    delete[] host_verify;
}

int main() {
    printf("Peer-to-Peer Communication Demonstration\n");
    printf("========================================\n\n");
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount < 2) {
        printf("This demonstration requires at least 2 GPUs. Found %d GPU(s).\n", deviceCount);
        return 1;
    }
    
    // Check P2P capabilities
    checkP2PCapabilities();
    
    // Enable P2P access
    enableP2PAccess(deviceCount);
    
    // Bandwidth measurements
    printf("=== Bandwidth Measurements ===\n");
    
    const size_t testSize = TRANSFER_SIZE;
    printf("Transfer size: %.2f MB\n", testSize / (1024.0*1024.0));
    printf("Number of iterations: %d\n\n", NUM_ITERATIONS);
    
    // Measure host-device bandwidth for comparison
    printf("Host-Device Bandwidth:\n");
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        double bandwidth = measureHostDeviceBandwidth(gpu, testSize);
        printf("  Host -> GPU %d: %.2f GB/s\n", gpu, bandwidth);
    }
    printf("\n");
    
    // Measure P2P bandwidth
    printf("P2P Bandwidth Matrix:\n");
    printf("From\\To  ");
    for (int j = 0; j < deviceCount; j++) {
        printf("GPU%d     ", j);
    }
    printf("\n");
    
    for (int i = 0; i < deviceCount; i++) {
        printf("GPU%d     ", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) {
                printf("----     ");
            } else {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                
                if (canAccessPeer) {
                    double bandwidth = measureP2PBandwidth(i, j, testSize);
                    printf("%.2f GB/s", bandwidth);
                } else {
                    printf("No P2P   ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");
    
    // Demonstrate asynchronous P2P transfers
    if (deviceCount >= 2) {
        demonstrateAsyncP2P(0, 1);
        printf("\n");
    }
    
    // Ring communication pattern
    if (deviceCount >= 3) {
        demonstrateRingCommunication(deviceCount);
        printf("\n");
    }
    
    // All-reduce operation
    if (deviceCount >= 2) {
        demonstrateAllReduce(deviceCount);
        printf("\n");
    }
    
    // Performance analysis
    printf("=== Performance Analysis ===\n");
    printf("Key Observations:\n");
    printf("1. P2P bandwidth varies significantly between GPU pairs\n");
    printf("2. NVLink provides much higher bandwidth than PCIe\n");
    printf("3. Asynchronous transfers can overlap with computation\n");
    printf("4. Communication patterns affect overall performance\n");
    
    printf("\nBest Practices:\n");
    printf("- Check P2P capabilities before enabling access\n");
    printf("- Use asynchronous transfers when possible\n");
    printf("- Consider topology when designing communication patterns\n");
    printf("- Minimize data movement between GPUs\n");
    printf("- Use streams to overlap communication and computation\n");
    
    printf("\nP2P Communication demonstration completed!\n");
    return 0;
}