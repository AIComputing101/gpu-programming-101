# Module 2: Advanced GPU Memory Management and Optimization
*Mastering GPU Memory Hierarchies and Performance Optimization*

> Environment note: Examples are tested in Docker containers with CUDA 12.9.1 (Ubuntu 22.04) and ROCm 7.0 (Ubuntu 24.04). The improved build system automatically optimizes memory access patterns. Prefer Docker for reproducible builds.

## Learning Objectives
After completing this module, you will be able to:
- Master GPU memory hierarchy and optimization strategies
- Implement efficient shared memory algorithms
- Optimize memory access patterns for maximum bandwidth
- Use texture and constant memory effectively
- Apply unified memory for simplified programming
- Profile and analyze memory performance bottlenecks
- Design memory-efficient parallel algorithms

## 2.1 GPU Memory Architecture Deep Dive

### Understanding Memory Hierarchy Performance

GPU memory systems are designed for high throughput rather than low latency. Understanding the performance characteristics of each memory type is crucial for optimization:

```
Memory Type       | Latency    | Bandwidth  | Size        | Scope
------------------|------------|------------|-------------|-------------
Registers         | 1 cycle    | ~20 TB/s   | ~256KB/SM   | Per-thread
Shared Memory     | ~20 cycles | ~15 TB/s   | 48-164KB/SM | Per-block
L1 Cache          | ~25 cycles | ~12 TB/s   | 128KB/SM    | Per-SM
L2 Cache          | ~200 cycles| ~3 TB/s    | 6-40MB      | Global
Global Memory     | ~300 cycles| ~1-3 TB/s  | 16-192GB    | Global
```

### Memory Access Patterns

**Coalesced Access (Efficient):**
```cuda
// Threads in a warp access consecutive memory locations
__global__ void coalescedAccess(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = tid;  // Perfect coalescing
}
```

**Strided Access (Inefficient):**
```cuda
// Threads access memory with large strides
__global__ void stridedAccess(float *data, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid * stride] = tid;  // Poor coalescing
}
```

**Random Access (Very Inefficient):**
```cuda
// Threads access random memory locations
__global__ void randomAccess(float *data, int *indices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[indices[tid]] = tid;  // No coalescing possible
}
```

## 2.2 Shared Memory Programming

### Shared Memory Basics

Shared memory is fast, on-chip memory shared among threads in a block. It's programmable and essential for high-performance GPU computing.

**Key Features:**
- ~15TB/s bandwidth (vs ~1-3TB/s for global memory)
- User-controlled caching
- Enables thread cooperation within blocks
- Limited size (48-164KB per SM depending on architecture)

### Matrix Transpose with Shared Memory

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

__global__ void matrixTransposeShared(float *input, float *output, 
                                     int width, int height) {
    // Shared memory tile
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    // Global indices
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Write transposed data to output
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // Swapped block indices
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Naive transpose for comparison
__global__ void matrixTransposeNaive(float *input, float *output, 
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}
```

### Bank Conflicts and Avoidance

Shared memory is organized into banks. Accessing the same bank simultaneously causes conflicts:

```cuda
__global__ void bankConflictExample() {
    __shared__ float shared_data[32][32];
    
    // Bank conflict: All threads access bank 0
    float value1 = shared_data[threadIdx.x][0];  // BAD
    
    // No bank conflict: Each thread accesses different bank
    float value2 = shared_data[threadIdx.x][threadIdx.x];  // GOOD
    
    // Avoid conflicts with padding
    __shared__ float padded_data[32][33];  // +1 padding
    float value3 = padded_data[0][threadIdx.x];  // GOOD
}
```

### Reduction with Shared Memory

```cuda
__global__ void reductionShared(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

## 2.3 Memory Coalescing Optimization

### Understanding Coalescing

Modern GPUs load memory in 128-byte transactions. For optimal performance, threads in a warp should access consecutive 4-byte words.

### Coalescing Analysis Tool

```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

class MemoryBenchmark {
private:
    float *d_data;
    size_t size;
    
public:
    MemoryBenchmark(size_t n) : size(n * sizeof(float)) {
        cudaMalloc(&d_data, size);
        
        // Initialize with pattern
        float *h_data = new float[n];
        for (size_t i = 0; i < n; i++) {
            h_data[i] = static_cast<float>(i);
        }
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        delete[] h_data;
    }
    
    ~MemoryBenchmark() {
        cudaFree(d_data);
    }
    
    float testCoalesced(int blocks, int threads) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        coalescedAccess<<<blocks, threads>>>(d_data, size / sizeof(float));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return time;
    }
    
    float testStrided(int blocks, int threads, int stride) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        stridedAccess<<<blocks, threads>>>(d_data, size / sizeof(float), stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return time;
    }
};

__global__ void coalescedAccess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void stridedAccess(float *data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### Structure of Arrays vs Array of Structures

```cuda
// Array of Structures (AoS) - Poor coalescing
struct Particle {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;
};

__global__ void updateParticlesAoS(Particle *particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Non-coalesced: threads access scattered memory
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

// Structure of Arrays (SoA) - Good coalescing
struct ParticlesSoA {
    float *x, *y, *z;       // Position arrays
    float *vx, *vy, *vz;    // Velocity arrays
    float *mass;
};

__global__ void updateParticlesSoA(ParticlesSoA particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced: threads access consecutive memory
        particles.x[idx] += particles.vx[idx];
        particles.y[idx] += particles.vy[idx];
        particles.z[idx] += particles.vz[idx];
    }
}
```

## 2.4 Texture and Constant Memory

### Texture Memory Benefits

Texture memory provides:
- Cached access with spatial locality optimization
- Automatic interpolation and filtering
- Bound checking with configurable border handling
- Read-only access optimized for 2D spatial locality

### Modern Texture Objects (CUDA)

```cuda
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

__global__ void textureKernel(cudaTextureObject_t texObj, float *output, 
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Texture coordinates are normalized [0,1]
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        // Fetch with automatic bilinear interpolation
        float value = tex2D<float>(texObj, u, v);
        
        output[y * width + x] = value;
    }
}

void setupTexture(float *h_data, int width, int height) {
    // Allocate CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    
    // Copy data to CUDA array
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, 
                       width * sizeof(float), width * sizeof(float), 
                       height, cudaMemcpyHostToDevice);
    
    // Specify texture resource
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    // Specify texture object parameters
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;  // Normalize coordinates [0,1]
    
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    float *d_output;
    cudaMalloc(&d_output, width * height * sizeof(float));
    
    textureKernel<<<gridSize, blockSize>>>(texObj, d_output, width, height);
    
    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
}
```

### Constant Memory Usage

```cuda
// Declare constant memory (max 64KB)
__constant__ float const_coefficients[1024];

__global__ void convolutionKernel(float *input, float *output, 
                                 int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
                
                // Fast access to constant memory
                sum += input[py * width + px] * const_coefficients[kernelIdx];
            }
        }
        
        output[y * width + x] = sum;
    }
}

void setupConstantMemory(float *h_kernel, int kernelSize) {
    // Copy to constant memory
    cudaMemcpyToSymbol(const_coefficients, h_kernel, 
                      kernelSize * kernelSize * sizeof(float));
}
```

## 2.5 Unified Memory and Memory Management

### Unified Memory Overview

Unified Memory provides a single memory space accessible by both CPU and GPU, with automatic migration managed by the CUDA runtime.

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

class UnifiedMemoryExample {
private:
    float *data;
    size_t size;
    
public:
    UnifiedMemoryExample(size_t n) : size(n) {
        // Allocate unified memory
        cudaMallocManaged(&data, n * sizeof(float));
        
        // Initialize on CPU
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<float>(i);
        }
    }
    
    ~UnifiedMemoryExample() {
        cudaFree(data);
    }
    
    void processOnGPU(int blocks, int threads) {
        // Launch kernel - data automatically migrates to GPU
        processData<<<blocks, threads>>>(data, size);
        
        // Wait for kernel to complete
        cudaDeviceSynchronize();
    }
    
    void processOnCPU() {
        // Access on CPU - data automatically migrates back
        for (size_t i = 0; i < size; i++) {
            data[i] *= 2.0f;
        }
    }
    
    void printResults(int count = 10) {
        printf("First %d elements: ", count);
        for (int i = 0; i < count && i < size; i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
};

__global__ void processData(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}
```

### Memory Prefetching and Hints

```cuda
void optimizedUnifiedMemory(float *data, size_t n, int device) {
    // Prefetch data to GPU before kernel launch
    cudaMemPrefetchAsync(data, n * sizeof(float), device);
    
    // Set memory usage hints
    cudaMemAdvise(data, n * sizeof(float), cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(data, n * sizeof(float), cudaMemAdviseSetPreferredLocation, device);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    processData<<<gridSize, blockSize>>>(data, n);
    
    // Prefetch back to CPU if needed
    cudaMemPrefetchAsync(data, n * sizeof(float), cudaCpuDeviceId);
}
```

### Memory Pool Management

```cuda
class MemoryPool {
private:
    cudaMemPool_t mempool;
    
public:
    MemoryPool() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        if (prop.memoryPoolsSupported) {
            cudaMemPoolProps poolProps = {};
            poolProps.allocType = cudaMemAllocationTypePinned;
            poolProps.handleTypes = cudaMemHandleTypeNone;
            poolProps.location.type = cudaMemLocationTypeDevice;
            poolProps.location.id = 0;
            
            cudaMemPoolCreate(&mempool, &poolProps);
        }
    }
    
    void* allocate(size_t size) {
        void *ptr;
        if (mempool) {
            cudaMallocFromPoolAsync(&ptr, size, mempool);
        } else {
            cudaMalloc(&ptr, size);
        }
        return ptr;
    }
    
    void deallocate(void *ptr) {
        cudaFreeAsync(ptr, 0);
    }
    
    ~MemoryPool() {
        if (mempool) {
            cudaMemPoolDestroy(mempool);
        }
    }
};
```

## 2.6 Performance Profiling and Analysis

### Memory Bandwidth Measurement

```cuda
#include <cuda_runtime.h>
#include <chrono>

class BandwidthBenchmark {
public:
    static double measureBandwidth(void (*kernel)(float*, size_t), 
                                  size_t elements, int iterations = 100) {
        float *d_data;
        size_t bytes = elements * sizeof(float);
        
        cudaMalloc(&d_data, bytes);
        
        // Warm up
        kernel(d_data, elements);
        cudaDeviceSynchronize();
        
        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            kernel(d_data, elements);
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double time = std::chrono::duration<double>(end - start).count();
        double bandwidth = (bytes * iterations) / (time * 1e9);  // GB/s
        
        cudaFree(d_data);
        return bandwidth;
    }
};

__global__ void copyKernel(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx];  // Simple copy
    }
}

__global__ void scaleKernel(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 1.5f;  // Scale operation
    }
}

void runBenchmarks() {
    size_t elements = 64 * 1024 * 1024;  // 64M elements
    
    auto copyBW = BandwidthBenchmark::measureBandwidth(copyKernel, elements);
    auto scaleBW = BandwidthBenchmark::measureBandwidth(scaleKernel, elements);
    
    printf("Copy bandwidth: %.2f GB/s\n", copyBW);
    printf("Scale bandwidth: %.2f GB/s\n", scaleBW);
}
```

### Memory Access Pattern Analysis

```cuda
__global__ void analyzeAccess(float *data, int *pattern, size_t n) {
    extern __shared__ int access_count[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory counters
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            access_count[i] = 0;
        }
    }
    __syncthreads();
    
    if (idx < n) {
        // Simulate access pattern
        int access_idx = pattern[idx] % blockDim.x;
        atomicAdd(&access_count[access_idx], 1);
        
        // Actual data access
        data[idx] = sqrtf(data[idx]);
    }
    
    __syncthreads();
    
    // Output access pattern statistics
    if (tid == 0) {
        printf("Block %d access pattern:\n", blockIdx.x);
        for (int i = 0; i < blockDim.x; i++) {
            if (access_count[i] > 0) {
                printf("  Thread %d: %d accesses\n", i, access_count[i]);
            }
        }
    }
}
```

## 2.7 Advanced Optimization Techniques

### Warp-Level Primitives

```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void warpReduceExample(float *input, float *output, size_t n) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += warp.shfl_down(value, offset);
    }
    
    // First thread in warp writes result
    if (warp.thread_rank() == 0) {
        output[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = value;
    }
}

__global__ void warpVoteExample(float *data, int *flags, size_t n) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool condition = (idx < n) && (data[idx] > 0.5f);
    
    // Warp vote operations
    bool any_positive = warp.any(condition);
    bool all_positive = warp.all(condition);
    unsigned ballot = warp.ballot(condition);
    
    if (warp.thread_rank() == 0) {
        flags[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = 
            (any_positive ? 1 : 0) + (all_positive ? 2 : 0) + __popc(ballot) * 4;
    }
}
```

### Memory Alignment and Vectorization

```cuda
// Aligned memory allocation
void* alignedAlloc(size_t size, size_t alignment) {
    void *ptr;
    cudaMalloc(&ptr, size + alignment);
    
    // Align to boundary
    size_t addr = reinterpret_cast<size_t>(ptr);
    size_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    
    return reinterpret_cast<void*>(aligned_addr);
}

// Vectorized memory operations
__global__ void vectorizedCopy(float4 *input, float4 *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Copy 16 bytes (4 floats) in one transaction
        output[idx] = input[idx];
    }
}

__global__ void vectorizedCompute(float4 *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float4 vec = data[idx];
        
        // Vectorized operations
        vec.x = sqrtf(vec.x);
        vec.y = sqrtf(vec.y);
        vec.z = sqrtf(vec.z);
        vec.w = sqrtf(vec.w);
        
        data[idx] = vec;
    }
}
```

## Summary

This module covered advanced GPU memory management and optimization:

- **Memory Hierarchy**: Understanding performance characteristics of different memory types
- **Shared Memory**: Implementing high-performance algorithms with thread cooperation
- **Memory Coalescing**: Optimizing access patterns for maximum bandwidth
- **Texture/Constant Memory**: Leveraging specialized memory types
- **Unified Memory**: Simplifying memory management with automatic migration
- **Performance Analysis**: Profiling and measuring memory performance
- **Advanced Techniques**: Warp-level operations and vectorization

**Key Takeaways:**
1. Memory bandwidth often limits GPU performance more than compute
2. Coalesced access patterns are crucial for high performance
3. Shared memory enables algorithm optimizations impossible with global memory
4. Modern GPUs provide sophisticated memory management features
5. Profiling is essential for identifying and fixing memory bottlenecks

**Next Steps:**
- Practice with the provided examples
- Profile your own kernels to identify memory bottlenecks
- Experiment with different memory access patterns
- Learn about advanced algorithms in Module 3