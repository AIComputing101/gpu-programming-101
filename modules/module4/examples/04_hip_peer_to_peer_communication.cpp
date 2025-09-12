#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

#define TRANSFER_SIZE (64 * 1024 * 1024)  // 64MB transfers
#define NUM_ITERATIONS 10
#define BLOCK_SIZE 256

// Simple kernel for generating data patterns
__global__ void generateData(float *data, int size, int pattern) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = sinf(idx * 0.001f + pattern) * cosf(idx * 0.0005f + pattern);
    }
}

// Kernel for processing data received from another GPU
__global__ void processReceivedData(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply some computation to the received data
        float value = input[idx];
        for (int i = 0; i < 10; i++) {
            value = sinf(value * 1.1f) + cosf(value * 0.9f);
        }
        output[idx] = value;
    }
}

// All-reduce kernel for multi-GPU reduction
__global__ void allReduceKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple scaling to demonstrate collective operation
        data[idx] *= 0.5f;
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

struct P2PInfo {
    int srcDevice;
    int dstDevice;
    bool canAccessPeer;
    double bandwidth;  // GB/s
};

class P2PManager {
private:
    int numGPUs;
    std::vector<std::vector<P2PInfo>> p2pMatrix;
    std::vector<hipStream_t> streams;
    
public:
    P2PManager() {
        HIP_CHECK(hipGetDeviceCount(&numGPUs));
        
        if (numGPUs < 2) {
            printf("P2P communication requires at least 2 GPUs. Found: %d\n", numGPUs);
            return;
        }
        
        printf("=== P2P Capability Detection ===\n");
        printf("Found %d GPU(s)\n\n", numGPUs);
        
        // Initialize P2P matrix
        p2pMatrix.resize(numGPUs);
        for (int i = 0; i < numGPUs; i++) {
            p2pMatrix[i].resize(numGPUs);
        }
        
        streams.resize(numGPUs);
        
        // Check P2P capabilities between all GPU pairs
        for (int src = 0; src < numGPUs; src++) {
            HIP_CHECK(hipSetDevice(src));
            HIP_CHECK(hipStreamCreate(&streams[src]));
            
            for (int dst = 0; dst < numGPUs; dst++) {
                p2pMatrix[src][dst].srcDevice = src;
                p2pMatrix[src][dst].dstDevice = dst;
                p2pMatrix[src][dst].bandwidth = 0.0;
                
                if (src == dst) {
                    p2pMatrix[src][dst].canAccessPeer = false;
                    continue;
                }
                
                int canAccessPeer;
                HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, src, dst));
                p2pMatrix[src][dst].canAccessPeer = (canAccessPeer == 1);
                
                if (canAccessPeer) {
                    // Enable P2P access
                    hipError_t result = hipDeviceEnablePeerAccess(dst, 0);
                    if (result == hipSuccess) {
                        printf("Enabled P2P access: GPU %d -> GPU %d\n", src, dst);
                    } else if (result == hipErrorPeerAccessAlreadyEnabled) {
                        printf("P2P access already enabled: GPU %d -> GPU %d\n", src, dst);
                    } else {
                        printf("Failed to enable P2P access: GPU %d -> GPU %d (%s)\n", 
                               src, dst, hipGetErrorString(result));
                        p2pMatrix[src][dst].canAccessPeer = false;
                    }
                } else {
                    printf("P2P access not supported: GPU %d -> GPU %d\n", src, dst);
                }
            }
        }
        printf("\n");
        
        // Print P2P capability matrix
        printP2PMatrix();
    }
    
    ~P2PManager() {
        for (int i = 0; i < numGPUs; i++) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamDestroy(streams[i]));
            
            // Disable P2P access
            for (int j = 0; j < numGPUs; j++) {
                if (i != j && p2pMatrix[i][j].canAccessPeer) {
                    hipDeviceDisablePeerAccess(j);
                }
            }
        }
    }
    
    void printP2PMatrix() {
        printf("P2P Access Matrix:\n");
        printf("     ");
        for (int i = 0; i < numGPUs; i++) {
            printf("GPU%d ", i);
        }
        printf("\n");
        
        for (int src = 0; src < numGPUs; src++) {
            printf("GPU%d ", src);
            for (int dst = 0; dst < numGPUs; dst++) {
                if (src == dst) {
                    printf("  -  ");
                } else {
                    printf(" %s ", p2pMatrix[src][dst].canAccessPeer ? "Y" : "N");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    
    int getNumGPUs() const { return numGPUs; }
    bool canAccess(int src, int dst) const { 
        return p2pMatrix[src][dst].canAccessPeer; 
    }
    hipStream_t getStream(int gpu) const { return streams[gpu]; }
    
    // Measure P2P bandwidth between two GPUs
    double measureBandwidth(int srcGPU, int dstGPU, size_t size) {
        if (!canAccess(srcGPU, dstGPU)) {
            return 0.0;
        }
        
        float *d_src, *d_dst;
        
        // Allocate memory on source GPU
        HIP_CHECK(hipSetDevice(srcGPU));
        HIP_CHECK(hipMalloc(&d_src, size));
        
        // Allocate memory on destination GPU
        HIP_CHECK(hipSetDevice(dstGPU));
        HIP_CHECK(hipMalloc(&d_dst, size));
        
        // Initialize source data
        HIP_CHECK(hipSetDevice(srcGPU));
        dim3 block(BLOCK_SIZE);
        dim3 grid((size/sizeof(float) + block.x - 1) / block.x);
        hipLaunchKernelGGL(generateData, grid, block, 0, streams[srcGPU], 
                          d_src, size/sizeof(float), srcGPU);
        HIP_CHECK(hipStreamSynchronize(streams[srcGPU]));
        
        // Warm up
        for (int i = 0; i < 3; i++) {
            HIP_CHECK(hipMemcpyPeerAsync(d_dst, dstGPU, d_src, srcGPU, size, streams[srcGPU]));
            HIP_CHECK(hipStreamSynchronize(streams[srcGPU]));
        }
        
        // Measure bandwidth
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            HIP_CHECK(hipMemcpyPeerAsync(d_dst, dstGPU, d_src, srcGPU, size, streams[srcGPU]));
        }
        HIP_CHECK(hipStreamSynchronize(streams[srcGPU]));
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double timeSeconds = duration.count() / 1e6;
        double bandwidth = (size * NUM_ITERATIONS) / timeSeconds / 1e9;  // GB/s
        
        // Cleanup
        HIP_CHECK(hipSetDevice(srcGPU));
        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipSetDevice(dstGPU));
        HIP_CHECK(hipFree(d_dst));
        
        return bandwidth;
    }
    
    // Measure all P2P bandwidths
    void measureAllBandwidths() {
        printf("=== P2P Bandwidth Measurement ===\n");
        printf("Transfer size: %.1f MB\n", TRANSFER_SIZE / (1024.0f * 1024.0f));
        printf("Iterations per measurement: %d\n\n", NUM_ITERATIONS);
        
        for (int src = 0; src < numGPUs; src++) {
            for (int dst = 0; dst < numGPUs; dst++) {
                if (src != dst && canAccess(src, dst)) {
                    double bandwidth = measureBandwidth(src, dst, TRANSFER_SIZE);
                    p2pMatrix[src][dst].bandwidth = bandwidth;
                    printf("GPU %d -> GPU %d: %.1f GB/s\n", src, dst, bandwidth);
                }
            }
        }
        printf("\n");
    }
    
    // Print bandwidth matrix
    void printBandwidthMatrix() {
        printf("P2P Bandwidth Matrix (GB/s):\n");
        printf("       ");
        for (int i = 0; i < numGPUs; i++) {
            printf("GPU%d   ", i);
        }
        printf("\n");
        
        for (int src = 0; src < numGPUs; src++) {
            printf("GPU%d  ", src);
            for (int dst = 0; dst < numGPUs; dst++) {
                if (src == dst) {
                    printf("  -    ");
                } else if (canAccess(src, dst)) {
                    printf("%5.1f  ", p2pMatrix[src][dst].bandwidth);
                } else {
                    printf(" N/A   ");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
};

// Compare P2P vs host-routed transfers
void compareTransferMethods(P2PManager& manager) {
    if (manager.getNumGPUs() < 2) return;
    
    printf("=== P2P vs Host-Routed Transfer Comparison ===\n");
    
    int srcGPU = 0;
    int dstGPU = 1;
    size_t size = TRANSFER_SIZE;
    
    if (!manager.canAccess(srcGPU, dstGPU)) {
        printf("P2P not available between GPU %d and GPU %d\n", srcGPU, dstGPU);
        return;
    }
    
    float *d_src, *d_dst, *h_temp;
    
    // Allocate device memory
    HIP_CHECK(hipSetDevice(srcGPU));
    HIP_CHECK(hipMalloc(&d_src, size));
    
    HIP_CHECK(hipSetDevice(dstGPU));
    HIP_CHECK(hipMalloc(&d_dst, size));
    
    // Allocate host memory for comparison
    HIP_CHECK(hipHostMalloc(&h_temp, size));
    
    // Initialize source data
    HIP_CHECK(hipSetDevice(srcGPU));
    dim3 block(BLOCK_SIZE);
    dim3 grid((size/sizeof(float) + block.x - 1) / block.x);
    hipLaunchKernelGGL(generateData, grid, block, 0, 0, d_src, size/sizeof(float), 0);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Measure P2P transfer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpyPeerAsync(d_dst, dstGPU, d_src, srcGPU, size, manager.getStream(srcGPU)));
    }
    HIP_CHECK(hipStreamSynchronize(manager.getStream(srcGPU)));
    auto end = std::chrono::high_resolution_clock::now();
    
    double p2pTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double p2pBandwidth = (size * NUM_ITERATIONS) / (p2pTime / 1000.0) / 1e9;
    
    // Measure host-routed transfer
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIP_CHECK(hipSetDevice(srcGPU));
        HIP_CHECK(hipMemcpy(h_temp, d_src, size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(dstGPU));
        HIP_CHECK(hipMemcpy(d_dst, h_temp, size, hipMemcpyHostToDevice));
    }
    end = std::chrono::high_resolution_clock::now();
    
    double hostTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double hostBandwidth = (size * NUM_ITERATIONS) / (hostTime / 1000.0) / 1e9;
    
    printf("Transfer size: %.1f MB\n", size / (1024.0f * 1024.0f));
    printf("P2P transfer: %.1f ms (%.1f GB/s)\n", p2pTime, p2pBandwidth);
    printf("Host-routed transfer: %.1f ms (%.1f GB/s)\n", hostTime, hostBandwidth);
    printf("P2P speedup: %.1fx\n\n", hostTime / p2pTime);
    
    // Cleanup
    HIP_CHECK(hipSetDevice(srcGPU));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipSetDevice(dstGPU));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipHostFree(h_temp));
}

// Demonstrate ring communication pattern
void demonstrateRingCommunication(P2PManager& manager) {
    if (manager.getNumGPUs() < 2) return;
    
    printf("=== Ring Communication Pattern ===\n");
    
    int numGPUs = manager.getNumGPUs();
    size_t elementCount = TRANSFER_SIZE / sizeof(float);
    size_t bytes = elementCount * sizeof(float);
    
    // Check if we can form a ring
    bool canFormRing = true;
    for (int i = 0; i < numGPUs; i++) {
        int next = (i + 1) % numGPUs;
        if (!manager.canAccess(i, next)) {
            canFormRing = false;
            break;
        }
    }
    
    if (!canFormRing) {
        printf("Cannot form complete ring - P2P not available between all adjacent GPUs\n\n");
        return;
    }
    
    printf("Forming ring with %d GPUs\n", numGPUs);
    
    // Allocate memory on each GPU
    std::vector<float*> d_data(numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&d_data[i], bytes));
        
        // Initialize data with GPU-specific pattern
        dim3 block(BLOCK_SIZE);
        dim3 grid((elementCount + block.x - 1) / block.x);
        hipLaunchKernelGGL(generateData, grid, block, 0, manager.getStream(i), 
                          d_data[i], elementCount, i + 1);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(manager.getStream(i)));
    }
    
    printf("Starting ring communication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform ring communication - each GPU sends to the next
    for (int step = 0; step < numGPUs; step++) {
        for (int gpu = 0; gpu < numGPUs; gpu++) {
            int nextGPU = (gpu + 1) % numGPUs;
            
            // Each GPU sends its data to the next GPU in the ring
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipMemcpyPeerAsync(d_data[nextGPU], nextGPU, 
                                        d_data[gpu], gpu, bytes, manager.getStream(gpu)));
        }
        
        // Synchronize all streams
        for (int gpu = 0; gpu < numGPUs; gpu++) {
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipStreamSynchronize(manager.getStream(gpu)));
        }
        
        printf("  Completed ring step %d/%d\n", step + 1, numGPUs);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ringTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    // Calculate total data transferred
    double totalData = bytes * numGPUs * numGPUs / 1e9;  // GB
    double effectiveBandwidth = totalData / (ringTime / 1000.0);
    
    printf("Ring communication completed\n");
    printf("Time: %.1f ms\n", ringTime);
    printf("Total data transferred: %.1f GB\n", totalData);
    printf("Effective bandwidth: %.1f GB/s\n\n", effectiveBandwidth);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(d_data[i]));
    }
}

// Demonstrate all-reduce operation
void demonstrateAllReduce(P2PManager& manager) {
    if (manager.getNumGPUs() < 2) return;
    
    printf("=== All-Reduce Communication Pattern ===\n");
    
    int numGPUs = manager.getNumGPUs();
    size_t elementCount = 1024 * 1024;  // 1M elements for reduction
    size_t bytes = elementCount * sizeof(float);
    
    printf("Performing all-reduce with %d GPUs (%zu elements)\n", numGPUs, elementCount);
    
    // Allocate memory on each GPU
    std::vector<float*> d_data(numGPUs);
    std::vector<float*> d_result(numGPUs);
    
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&d_data[i], bytes));
        HIP_CHECK(hipMalloc(&d_result[i], bytes));
        
        // Initialize data
        dim3 block(BLOCK_SIZE);
        dim3 grid((elementCount + block.x - 1) / block.x);
        hipLaunchKernelGGL(generateData, grid, block, 0, manager.getStream(i), 
                          d_data[i], elementCount, i + 1);
        
        // Initialize result
        HIP_CHECK(hipMemsetAsync(d_result[i], 0, bytes, manager.getStream(i)));
    }
    
    // Synchronize initialization
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(manager.getStream(i)));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simple all-reduce implementation: gather on GPU 0, then broadcast
    
    // Phase 1: Gather all data on GPU 0
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpyAsync(d_result[0], d_data[0], bytes, hipMemcpyDeviceToDevice, manager.getStream(0)));
    
    for (int gpu = 1; gpu < numGPUs; gpu++) {
        if (manager.canAccess(gpu, 0)) {
            // Direct P2P copy to GPU 0
            HIP_CHECK(hipMemcpyPeerAsync(d_result[0], 0, d_data[gpu], gpu, bytes, manager.getStream(0)));
        } else {
            // Fall back to host-routed transfer
            float *h_temp;
            HIP_CHECK(hipHostMalloc(&h_temp, bytes));
            
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipMemcpy(h_temp, d_data[gpu], bytes, hipMemcpyDeviceToHost));
            
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMemcpy(d_result[0], h_temp, bytes, hipMemcpyHostToDevice));
            
            HIP_CHECK(hipHostFree(h_temp));
        }
        
        // Perform reduction on GPU 0
        dim3 block(BLOCK_SIZE);
        dim3 grid((elementCount + block.x - 1) / block.x);
        hipLaunchKernelGGL(allReduceKernel, grid, block, 0, manager.getStream(0), 
                          d_result[0], elementCount);
    }
    
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(manager.getStream(0)));
    
    // Phase 2: Broadcast result to all GPUs
    for (int gpu = 1; gpu < numGPUs; gpu++) {
        if (manager.canAccess(0, gpu)) {
            HIP_CHECK(hipMemcpyPeerAsync(d_result[gpu], gpu, d_result[0], 0, bytes, manager.getStream(0)));
        } else {
            // Fall back to host-routed transfer
            float *h_temp;
            HIP_CHECK(hipHostMalloc(&h_temp, bytes));
            
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipMemcpy(h_temp, d_result[0], bytes, hipMemcpyDeviceToHost));
            
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipMemcpy(d_result[gpu], h_temp, bytes, hipMemcpyHostToDevice));
            
            HIP_CHECK(hipHostFree(h_temp));
        }
    }
    
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamSynchronize(manager.getStream(0)));
    
    auto end = std::chrono::high_resolution_clock::now();
    double allReduceTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    printf("All-reduce completed in %.1f ms\n", allReduceTime);
    printf("Effective bandwidth: %.1f GB/s\n\n", 
           (bytes * (numGPUs + 1)) / (allReduceTime / 1000.0) / 1e9);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_result[i]));
    }
}

void printSystemTopology() {
    printf("=== GPU System Topology ===\n");
    
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    for (int device = 0; device < deviceCount; device++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));
        
        printf("GPU %d: %s\n", device, prop.name);
        printf("  PCIe Domain:Bus:Device.Function: %04x:%02x:%02x.%x\n", 
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Memory Clock: %.1f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("\n");
    }
}

int main() {
    printf("=== HIP Peer-to-Peer Communication Demo ===\n\n");
    
    // Print system topology
    printSystemTopology();
    
    // Initialize P2P manager
    P2PManager manager;
    
    if (manager.getNumGPUs() < 2) {
        printf("This demo requires at least 2 GPUs\n");
        return 1;
    }
    
    // Measure P2P bandwidths
    manager.measureAllBandwidths();
    manager.printBandwidthMatrix();
    
    // Compare P2P vs host-routed transfers
    compareTransferMethods(manager);
    
    // Demonstrate communication patterns
    demonstrateRingCommunication(manager);
    demonstrateAllReduce(manager);
    
    printf("=== Performance Summary ===\n");
    printf("P2P communication enables:\n");
    printf("- Direct GPU-to-GPU memory transfers\n");
    printf("- Reduced host CPU involvement\n");
    printf("- Higher bandwidth for multi-GPU applications\n");
    printf("- Efficient collective communication patterns\n");
    printf("- Better scaling for distributed GPU computing\n\n");
    
    printf("Key HIP P2P features:\n");
    printf("- hipDeviceCanAccessPeer() - Check P2P capability\n");
    printf("- hipDeviceEnablePeerAccess() - Enable P2P access\n");
    printf("- hipMemcpyPeer() / hipMemcpyPeerAsync() - Direct transfers\n");
    printf("- Works across AMD and NVIDIA GPUs with compatible drivers\n");
    printf("- Topology-aware programming for optimal performance\n\n");
    
    printf("=== HIP P2P Communication Demo Completed Successfully ===\n");
    return 0;
}