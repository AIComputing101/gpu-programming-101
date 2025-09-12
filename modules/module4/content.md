# Module 4: Advanced GPU Programming - Multi-GPU, Streams, and Scalability

## Overview

This module covers advanced GPU programming techniques for maximizing performance and scalability across multiple GPUs, asynchronous execution with streams, unified memory management, and dynamic parallelism. These concepts are essential for building high-performance applications that can scale across modern GPU clusters and data centers.

## Learning Objectives

By the end of this module, you will:

- Master CUDA streams for asynchronous execution and overlapping computation with data transfers
- Implement multi-GPU applications with proper load balancing and synchronization
- Utilize unified memory for simplified memory management across CPU and GPU
- Implement peer-to-peer communication between multiple GPUs
- Apply dynamic parallelism for recursive and adaptive algorithms
- Optimize applications for maximum throughput and scalability
- Profile and debug multi-GPU applications effectively

## Core Concepts

### 1. CUDA Streams and Asynchronous Execution

#### What are CUDA Streams?

CUDA streams provide a way to execute multiple operations asynchronously on the GPU. A stream is a sequence of operations that execute in order on the GPU, but different streams can execute concurrently.

**Benefits of Streams:**
- Overlap computation with memory transfers
- Achieve higher GPU utilization
- Hide memory transfer latency
- Enable concurrent kernel execution

#### Stream Types

```cuda
// Default stream (stream 0) - synchronizing
// All operations wait for previous operations to complete

// Non-default streams - asynchronous
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
```

#### Asynchronous Memory Transfers

```cuda
// Asynchronous memory copy
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);

// Kernel launch on specific stream
kernel<<<grid, block, 0, stream1>>>(d_data);

// Asynchronous copy back
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream1);
```

#### Stream Synchronization

```cuda
// Wait for specific stream
cudaStreamSynchronize(stream1);

// Wait for all streams
cudaDeviceSynchronize();

// Check if stream is complete (non-blocking)
cudaError_t result = cudaStreamQuery(stream1);
```

### 2. Multi-GPU Programming

#### Device Management

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

// Set active device
cudaSetDevice(0);

// Get device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);
```

#### Multi-GPU Data Distribution

```cuda
// Distribute data across multiple GPUs
void distributeData(float *data, int totalSize, int numGPUs) {
    int chunkSize = totalSize / numGPUs;
    
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        
        float *d_data;
        cudaMalloc(&d_data, chunkSize * sizeof(float));
        
        cudaMemcpy(d_data, &data[gpu * chunkSize], 
                   chunkSize * sizeof(float), cudaMemcpyHostToDevice);
    }
}
```

#### Load Balancing Strategies

1. **Equal Division**: Simple split across GPUs
2. **Weighted Division**: Based on GPU compute capability
3. **Dynamic Load Balancing**: Runtime adjustment based on performance
4. **Work Stealing**: GPUs take work from busy neighbors

### 3. Unified Memory

#### What is Unified Memory?

Unified Memory provides a single memory space accessible by both CPU and GPU, simplifying memory management and enabling automatic data migration.

```cuda
// Allocate unified memory
float *data;
cudaMallocManaged(&data, size);

// Use on CPU
for (int i = 0; i < n; i++) {
    data[i] = i * 2.0f;
}

// Use on GPU
kernel<<<grid, block>>>(data, n);
```

#### Memory Access Patterns

```cuda
// Prefetch data to GPU
cudaMemPrefetchAsync(data, size, deviceId);

// Provide memory access hints
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);
```

#### Unified Memory Best Practices

1. **Use Memory Hints**: Guide the runtime system
2. **Prefetch Data**: Overlap migration with computation
3. **Avoid Fine-Grained Access**: Minimize CPU-GPU ping-pong
4. **Page Fault Awareness**: Understand migration overhead

### 4. Peer-to-Peer Communication

#### P2P Memory Access

```cuda
// Check P2P access capability
int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, device0, device1);

if (canAccessPeer) {
    // Enable P2P access
    cudaSetDevice(device0);
    cudaDeviceEnablePeerAccess(device1, 0);
    
    // Direct memory access between GPUs
    cudaMemcpyPeer(dst_ptr, device1, src_ptr, device0, size);
}
```

#### P2P Communication Patterns

```cuda
// All-to-All communication
void allToAllP2P(float **gpu_data, int numGPUs, int chunkSize) {
    for (int src = 0; src < numGPUs; src++) {
        cudaSetDevice(src);
        for (int dst = 0; dst < numGPUs; dst++) {
            if (src != dst) {
                cudaMemcpyPeerAsync(gpu_data[dst], dst, 
                                   gpu_data[src], src, 
                                   chunkSize, stream[src]);
            }
        }
    }
}
```

### 5. Dynamic Parallelism

#### GPU Kernel Launching Kernels

Dynamic parallelism allows GPU kernels to launch other kernels directly, enabling recursive algorithms and adaptive workload distribution.

```cuda
__global__ void parentKernel(float *data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0 && size > threshold) {
        // Launch child kernel from GPU
        dim3 childGrid(size/256);
        dim3 childBlock(256);
        
        childKernel<<<childGrid, childBlock>>>(data, size/2);
    }
}
```

#### Recursive Algorithms

```cuda
__global__ void quicksort(float *data, int left, int right) {
    if (left < right) {
        int pivot = partition(data, left, right);
        
        if (right - left > MIN_SIZE) {
            // Launch recursive kernels
            quicksort<<<1, 1>>>(data, left, pivot - 1);
            quicksort<<<1, 1>>>(data, pivot + 1, right);
            
            // Wait for child kernels
            cudaDeviceSynchronize();
        } else {
            // Small arrays: sort sequentially
            sequentialSort(data, left, right);
        }
    }
}
```

### 6. Performance Optimization Strategies

#### Stream Optimization

```cuda
// Pipeline execution across streams
for (int i = 0; i < numChunks; i++) {
    int streamId = i % numStreams;
    
    // Memory transfer
    cudaMemcpyAsync(d_input[streamId], h_input + i * chunkSize,
                    chunkSize, cudaMemcpyHostToDevice, streams[streamId]);
    
    // Kernel execution
    processData<<<grid, block, 0, streams[streamId]>>>(
        d_input[streamId], d_output[streamId], chunkSize);
    
    // Result transfer
    cudaMemcpyAsync(h_output + i * chunkSize, d_output[streamId],
                    chunkSize, cudaMemcpyDeviceToHost, streams[streamId]);
}
```

#### Multi-GPU Scaling

```cuda
// Measure scaling efficiency
void measureScaling(int numGPUs) {
    double singleGPUTime = runOnSingleGPU();
    double multiGPUTime = runOnMultipleGPUs(numGPUs);
    
    double speedup = singleGPUTime / multiGPUTime;
    double efficiency = speedup / numGPUs;
    
    printf("Speedup: %.2fx, Efficiency: %.1f%%\n", 
           speedup, efficiency * 100);
}
```

## Advanced Topics

### 1. NCCL (NVIDIA Collective Communications Library)

```cuda
#include <nccl.h>

// Initialize NCCL
ncclComm_t comm;
ncclUniqueId id;
ncclGetUniqueId(&id);
ncclCommInitRank(&comm, numGPUs, id, rank);

// All-reduce operation
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
```

### 2. Multi-Process Service (MPS)

MPS allows multiple processes to share a single GPU context, improving utilization for applications with small kernels.

```bash
# Start MPS daemon
nvidia-cuda-mps-control -d

# Run multiple processes
./app1 & ./app2 & ./app3 &

# Stop MPS
echo quit | nvidia-cuda-mps-control
```

### 3. NVLink and GPU Direct

```cuda
// Check NVLink connectivity
cudaDeviceP2PAttr attr;
cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrAccessSupported, 
                          device0, device1);

// GPU Direct RDMA for network communication
// Allows direct GPU-to-network transfers
```

### 4. Cooperative Groups Multi-GPU

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void multiGPUKernel(float *data) {
    auto grid = cg::this_multi_grid();
    
    // Operations across multiple GPUs
    float result = cg::reduce(grid, threadValue, cg::plus<float>());
}
```

## Memory Management Strategies

### 1. Memory Pools

```cuda
// Create memory pool
cudaMemPool_t mempool;
cudaMemPoolCreate(&mempool, &prop);

// Allocate from pool
void *ptr;
cudaMallocFromPoolAsync(&ptr, size, mempool, stream);
```

### 2. Memory Mapping

```cuda
// Map host memory for GPU access
float *hostPtr;
cudaHostAlloc(&hostPtr, size, cudaHostAllocMapped);

// Get device pointer
float *devicePtr;
cudaHostGetDevicePointer(&devicePtr, hostPtr, 0);
```

### 3. Async Memory Operations

```cuda
// Async memory set
cudaMemsetAsync(ptr, value, size, stream);

// 2D/3D memory copies
cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, 
                  kind, stream);
```

## Profiling and Debugging

### 1. Multi-GPU Profiling

```bash
# Profile multi-GPU application
nvprof --print-gpu-trace ./multi_gpu_app

# Nsight Systems for timeline analysis
nsys profile --trace=cuda,nvtx ./app
```

### 2. Stream Analysis

```bash
# Analyze stream utilization
nvprof --print-api-trace --log-file output.log ./app
```

### 3. P2P Bandwidth Testing

```cuda
void testP2PBandwidth(int device0, int device1) {
    // Measure P2P bandwidth between devices
    float *d_data0, *d_data1;
    
    cudaSetDevice(device0);
    cudaMalloc(&d_data0, size);
    
    cudaSetDevice(device1);
    cudaMalloc(&d_data1, size);
    
    // Time P2P transfers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpyPeer(d_data1, device1, d_data0, device0, size);
    cudaEventRecord(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    double bandwidth = (size / (1024.0*1024.0*1024.0)) / (time / 1000.0);
    printf("P2P Bandwidth: %.2f GB/s\n", bandwidth);
}
```

## Common Patterns and Best Practices

### 1. Pipeline Processing

```cuda
void pipelineProcessing(float *data, int totalSize) {
    const int numStreams = 4;
    const int chunkSize = totalSize / numStreams;
    
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Pipeline stages
    for (int chunk = 0; chunk < totalSize / chunkSize; chunk++) {
        int streamId = chunk % numStreams;
        
        // Stage 1: Transfer to GPU
        cudaMemcpyAsync(d_input, data + chunk * chunkSize,
                       chunkSize, cudaMemcpyHostToDevice, streams[streamId]);
        
        // Stage 2: Process on GPU
        processKernel<<<grid, block, 0, streams[streamId]>>>(d_input, d_output);
        
        // Stage 3: Transfer back
        cudaMemcpyAsync(results + chunk * chunkSize, d_output,
                       chunkSize, cudaMemcpyDeviceToHost, streams[streamId]);
    }
}
```

### 2. Multi-GPU Reduction

```cuda
void multiGPUReduction(float *data, int size, int numGPUs) {
    float *partial_results[numGPUs];
    
    // Phase 1: Reduce on each GPU
    #pragma omp parallel for
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        
        int chunkSize = size / numGPUs;
        float *d_chunk, *d_result;
        
        cudaMalloc(&d_chunk, chunkSize * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));
        
        cudaMemcpy(d_chunk, &data[gpu * chunkSize], 
                   chunkSize * sizeof(float), cudaMemcpyHostToDevice);
        
        reduceKernel<<<grid, block>>>(d_chunk, d_result, chunkSize);
        
        cudaMemcpy(&partial_results[gpu], d_result, sizeof(float), 
                   cudaMemcpyDeviceToHost);
    }
    
    // Phase 2: Final reduction on CPU
    float final_result = 0.0f;
    for (int i = 0; i < numGPUs; i++) {
        final_result += partial_results[i];
    }
}
```

### 3. Dynamic Load Balancing

```cuda
struct WorkQueue {
    int *tasks;
    int *head;
    int *tail;
    int capacity;
};

__global__ void dynamicWorkStealing(WorkQueue *queue, float *data) {
    __shared__ int local_task;
    
    while (true) {
        // Try to get work
        if (threadIdx.x == 0) {
            local_task = atomicAdd(queue->head, 1);
        }
        __syncthreads();
        
        if (local_task >= queue->capacity) break;
        
        int task_id = queue->tasks[local_task];
        processTask(data, task_id);
        
        __syncthreads();
    }
}
```

## Error Handling and Robustness

### 1. Multi-GPU Error Handling

```cuda
cudaError_t launchMultiGPU(int numGPUs) {
    cudaError_t result = cudaSuccess;
    
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        
        kernel<<<grid, block>>>(data);
        cudaError_t error = cudaGetLastError();
        
        if (error != cudaSuccess) {
            printf("GPU %d error: %s\n", gpu, cudaGetErrorString(error));
            result = error;
        }
    }
    
    return result;
}
```

### 2. Stream Error Handling

```cuda
void checkStreamErrors(cudaStream_t *streams, int numStreams) {
    for (int i = 0; i < numStreams; i++) {
        cudaError_t error = cudaStreamQuery(streams[i]);
        if (error != cudaSuccess && error != cudaErrorNotReady) {
            printf("Stream %d error: %s\n", i, cudaGetErrorString(error));
        }
    }
}
```

## Performance Considerations

### 1. Memory Bandwidth Utilization

- **Coalesced Access**: Ensure efficient memory access patterns
- **Stream Parallelism**: Overlap transfers with computation
- **P2P Communication**: Use fastest interconnect available

### 2. Load Balancing

- **Work Distribution**: Balance computation across GPUs
- **Communication Minimization**: Reduce inter-GPU data transfers
- **Synchronization Points**: Minimize global synchronization

### 3. Scaling Metrics

```cuda
// Amdahl's Law for parallel scaling
double parallelFraction = 0.95;  // 95% parallelizable
double maxSpeedup = 1.0 / (1.0 - parallelFraction + parallelFraction / numGPUs);

// Gustafson's Law for problem scaling
double scaledSpeedup = numGPUs - (1.0 - parallelFraction) * (numGPUs - 1);
```

## Summary

Module 4 covers the most advanced aspects of GPU programming, enabling you to build scalable applications that can efficiently utilize multiple GPUs and maximize system throughput. Key takeaways include:

1. **Streams enable asynchronous execution** and overlapping of computation with memory transfers
2. **Multi-GPU programming** requires careful attention to load balancing and communication
3. **Unified Memory** simplifies programming but requires understanding of migration costs
4. **Dynamic Parallelism** enables recursive and adaptive algorithms on the GPU
5. **Performance optimization** is critical for achieving good scaling across multiple devices

These techniques are essential for building high-performance applications in scientific computing, machine learning, and other computationally intensive domains that require the processing power of multiple GPUs working in concert.