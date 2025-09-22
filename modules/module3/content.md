# Module 3: Advanced GPU Algorithms and Parallel Patterns
*Mastering High-Performance Parallel Computing Algorithms*

> Environment note: Use the provided Docker images (CUDA 13.0.1 on Ubuntu 24.04, ROCm 7.0.1 on Ubuntu 24.04) with automatic GPU detection for consistent toolchains across platforms.

## Learning Objectives
After completing this module, you will be able to:
- Implement efficient reduction and scan algorithms
- Design and optimize parallel sorting algorithms
- Master convolution and stencil computation patterns
- Apply graph algorithms on GPU architectures
- Utilize cooperative groups for advanced thread coordination
- Analyze algorithm complexity and scalability on GPUs
- Design custom parallel algorithms for domain-specific problems

## 3.1 Introduction to Parallel Algorithm Design

### Fundamental Parallel Patterns

GPU computing excels at implementing fundamental parallel patterns that form the building blocks of complex algorithms:

```
Pattern Type          | Use Cases                    | Key Characteristics
--------------------|------------------------------|--------------------
Map                 | Element-wise operations      | Embarrassingly parallel
Reduce              | Sum, min/max, aggregations   | Tree-based combining
Scan (Prefix Sum)   | Cumulative operations        | Dependencies require care
Scatter/Gather      | Irregular memory access      | Data reorganization
Stencil             | Nearest-neighbor operations  | Spatial locality
Sort                | Data ordering                | Complex dependencies
```

### Algorithm Complexity on GPUs

Traditional CPU algorithm analysis must be adapted for GPU architectures:

**Time Complexity Considerations:**
- **Work**: Total number of operations (similar to sequential)
- **Depth**: Number of parallel steps (critical for GPUs)
- **Memory Access**: Often dominates execution time

**Space Complexity:**
- **Shared Memory**: Fast but limited (48-164KB per SM)
- **Global Memory**: Large but slow (GB-TB)
- **Register Usage**: Affects occupancy

### CUDA vs HIP Algorithm Implementation

Both CUDA and HIP support the same fundamental patterns with platform-specific optimizations:

```cpp
// CUDA approach
__global__ void algorithm_cuda(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Algorithm implementation
}

// HIP approach (cross-platform)
__global__ void algorithm_hip(float *data, int n) {
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // Same algorithm, different intrinsics
}
```

## 3.2 Reduction Algorithms

Reduction operations aggregate data across threads, blocks, or the entire dataset. Common reductions include sum, min/max, and logical operations.

### Basic Reduction Pattern

```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Simple block-level reduction
__global__ void blockReduce(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Optimized Warp-Level Reduction

```cuda
// Modern warp-level reduction using shuffle operations
__global__ void warpReduce(float *input, float *output, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += warp.shfl_down(value, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32]; // Max 32 warps per block
    if (warp.thread_rank() == 0) {
        warp_sums[warp.meta_group_rank()] = value;
    }
    
    __syncthreads();
    
    // Final reduction of warp sums
    if (threadIdx.x < block.size() / 32) {
        value = warp_sums[threadIdx.x];
    } else {
        value = 0.0f;
    }
    
    // Reduce final warp
    if (warp.meta_group_rank() == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += warp.shfl_down(value, offset);
        }
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = value;
        }
    }
}
```

### Multi-Pass Reduction Strategy

Large datasets require multiple kernel launches:

```cuda
class MultiPassReduction {
private:
    float *d_temp;
    size_t temp_size;
    
public:
    MultiPassReduction(size_t max_elements) {
        // Allocate temporary storage for intermediate results
        temp_size = (max_elements + 255) / 256; // Max blocks needed
        cudaMalloc(&d_temp, temp_size * sizeof(float));
    }
    
    float reduce(float *input, size_t n) {
        float *current_input = input;
        size_t current_n = n;
        float *current_output = d_temp;
        
        // Multi-pass reduction
        while (current_n > 1) {
            int threads = 256;
            int blocks = (current_n + threads - 1) / threads;
            
            if (blocks == 1) {
                // Final pass - use shared memory reduction
                blockReduce<<<1, threads>>>(current_input, current_output, current_n);
                break;
            } else {
                // Intermediate pass - use warp reduction
                warpReduce<<<blocks, threads>>>(current_input, current_output, current_n);
                
                current_input = current_output;
                current_n = blocks;
            }
        }
        
        // Copy final result back to host
        float result;
        cudaMemcpy(&result, current_output, sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
};
```

## 3.3 Scan (Prefix Sum) Algorithms

Scan operations compute cumulative results: each output element contains the reduction of all previous input elements.

### Inclusive vs Exclusive Scan

```cuda
// Inclusive scan: out[i] = in[0] + in[1] + ... + in[i]
// Exclusive scan: out[i] = in[0] + in[1] + ... + in[i-1]

__global__ void naiveScan(float *input, float *output, int n, bool inclusive) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float sum = 0.0f;
        int end = inclusive ? tid + 1 : tid;
        
        for (int i = 0; i < end; i++) {
            sum += input[i];
        }
        
        output[tid] = sum;
    }
}
```

### Efficient Block-Level Scan (Hillis-Steele)

```cuda
__global__ void blockScan(float *input, float *output, int n) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase (Hillis-Steele algorithm)
    for (int d = 1; d < blockDim.x; d <<= 1) {
        float val = 0.0f;
        if (tid >= d) {
            val = temp[tid - d];
        }
        __syncthreads();
        
        if (tid >= d) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    // Write result
    if (i < n) {
        output[i] = temp[tid];
    }
}
```

### Work-Efficient Scan (Blelloch)

```cuda
__global__ void workEfficientScan(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            temp[bi] += temp[ai];
        }
        __syncthreads();
    }
    
    // Clear last element (exclusive scan)
    if (tid == 0) {
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d <<= 1) {
        if (tid < d) {
            int ai = (2 * tid + 1) * (blockDim.x / d) - 1;
            int bi = (2 * tid + 2) * (blockDim.x / d) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }
    
    // Write result
    if (i < n) {
        output[i] = temp[tid];
    }
}
```

## 3.4 Sorting Algorithms

GPU sorting algorithms must balance parallelism with data dependencies.

### Parallel Bitonic Sort

```cuda
// Bitonic sorting network - always sorts 2^k elements
__global__ void bitonicSort(float *data, int n, int k, int j) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j; // XOR operation
    
    if (ixj > tid && tid < n && ixj < n) {
        // Determine sort direction based on position in bitonic sequence
        bool ascending = ((tid & k) == 0);
        
        if ((data[tid] > data[ixj]) == ascending) {
            // Swap elements
            float temp = data[tid];
            data[tid] = data[ixj];
            data[ixj] = temp;
        }
    }
}

void launchBitonicSort(float *d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Find next power of 2
    int n_pow2 = 1;
    while (n_pow2 < n) n_pow2 <<= 1;
    
    // Bitonic sort phases
    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSort<<<blocks, threads>>>(d_data, n, k, j);
            cudaDeviceSynchronize();
        }
    }
}
```

### Radix Sort Implementation

```cuda
// Radix sort using parallel prefix sum
__global__ void radixSortPass(unsigned int *input, unsigned int *output, 
                             int *indices, int n, int bit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned int value = input[tid];
        int bin = (value >> bit) & 1; // Extract bit
        
        // Store in appropriate bin using atomic operations
        int pos = atomicAdd(&indices[bin], 1);
        output[pos] = value;
    }
}

class RadixSort {
private:
    unsigned int *d_temp;
    int *d_indices;
    
public:
    void sort(unsigned int *d_data, int n) {
        const int bits = sizeof(unsigned int) * 8;
        
        for (int bit = 0; bit < bits; bit++) {
            // Reset counters
            cudaMemset(d_indices, 0, 2 * sizeof(int));
            
            // Count elements for each bin
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            
            radixSortPass<<<blocks, threads>>>(d_data, d_temp, d_indices, n, bit);
            cudaDeviceSynchronize();
            
            // Swap buffers
            std::swap(d_data, d_temp);
        }
    }
};
```

## 3.5 Convolution and Stencil Patterns

Stencil computations apply the same operation to every point in a grid using neighboring values.

### 1D Stencil with Shared Memory

```cuda
__global__ void stencil1D(float *input, float *output, int n) {
    const int RADIUS = 3;
    const int BLOCK_SIZE = 256;
    
    __shared__ float shared[BLOCK_SIZE + 2 * RADIUS];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with halos
    int shared_idx = tid + RADIUS;
    
    // Main data
    if (gid < n) {
        shared[shared_idx] = input[gid];
    } else {
        shared[shared_idx] = 0.0f;
    }
    
    // Left halo
    if (tid < RADIUS) {
        int left_idx = gid - RADIUS;
        shared[tid] = (left_idx >= 0) ? input[left_idx] : 0.0f;
    }
    
    // Right halo
    if (tid < RADIUS) {
        int right_idx = gid + BLOCK_SIZE;
        shared[shared_idx + BLOCK_SIZE] = (right_idx < n) ? input[right_idx] : 0.0f;
    }
    
    __syncthreads();
    
    // Apply stencil
    if (gid < n) {
        float result = 0.0f;
        for (int i = -RADIUS; i <= RADIUS; i++) {
            result += shared[shared_idx + i] * 0.1f; // Simple averaging
        }
        output[gid] = result;
    }
}
```

### 2D Convolution with Separable Filters

```cuda
// Separable convolution: 2D filter = 1D row filter * 1D column filter
__global__ void convolutionRows(float *input, float *output, int width, int height) {
    const int KERNEL_RADIUS = 8;
    const float kernel[17] = { /* Gaussian kernel coefficients */ };
    
    __shared__ float shared_data[256 + 2 * KERNEL_RADIUS];
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory
    int shared_idx = tid + KERNEL_RADIUS;
    int global_idx = row * width + col;
    
    if (col < width) {
        shared_data[shared_idx] = input[global_idx];
    } else {
        shared_data[shared_idx] = 0.0f;
    }
    
    // Load halo regions
    if (tid < KERNEL_RADIUS) {
        // Left halo
        int left_col = col - KERNEL_RADIUS;
        if (left_col >= 0) {
            shared_data[tid] = input[row * width + left_col];
        } else {
            shared_data[tid] = 0.0f;
        }
        
        // Right halo
        int right_col = col + blockDim.x;
        if (right_col < width) {
            shared_data[shared_idx + blockDim.x] = input[row * width + right_col];
        } else {
            shared_data[shared_idx + blockDim.x] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Apply convolution
    if (col < width) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            sum += shared_data[shared_idx + k] * kernel[k + KERNEL_RADIUS];
        }
        output[global_idx] = sum;
    }
}
```

## 3.6 Graph Algorithms on GPUs

Graph algorithms present unique challenges due to irregular memory access patterns and data dependencies.

### Breadth-First Search (BFS)

```cuda
// Level-synchronous BFS
__global__ void bfs_kernel(int *graph, int *offsets, int *levels, 
                          bool *frontier, bool *new_frontier, 
                          int num_vertices, int current_level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && frontier[tid]) {
        // Process all neighbors of vertex tid
        int start = offsets[tid];
        int end = offsets[tid + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = graph[i];
            
            // If neighbor hasn't been visited
            if (levels[neighbor] == -1) {
                levels[neighbor] = current_level + 1;
                new_frontier[neighbor] = true;
            }
        }
        
        frontier[tid] = false; // Remove from current frontier
    }
}

void gpu_bfs(int *h_graph, int *h_offsets, int num_vertices, int num_edges, int source) {
    // Allocate GPU memory
    int *d_graph, *d_offsets, *d_levels;
    bool *d_frontier, *d_new_frontier;
    
    cudaMalloc(&d_graph, num_edges * sizeof(int));
    cudaMalloc(&d_offsets, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_levels, num_vertices * sizeof(int));
    cudaMalloc(&d_frontier, num_vertices * sizeof(bool));
    cudaMalloc(&d_new_frontier, num_vertices * sizeof(bool));
    
    // Initialize
    cudaMemcpy(d_graph, h_graph, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Set initial conditions
    cudaMemset(d_levels, -1, num_vertices * sizeof(int));
    cudaMemset(d_frontier, false, num_vertices * sizeof(bool));
    
    // Set source
    int zero = 0;
    bool true_val = true;
    cudaMemcpy(&d_levels[source], &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_frontier[source], &true_val, sizeof(bool), cudaMemcpyHostToDevice);
    
    int current_level = 0;
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    
    // BFS iterations
    bool h_continue = true;
    while (h_continue) {
        cudaMemset(d_new_frontier, false, num_vertices * sizeof(bool));
        
        bfs_kernel<<<blocks, threads>>>(d_graph, d_offsets, d_levels, 
                                       d_frontier, d_new_frontier, 
                                       num_vertices, current_level);
        
        // Check if any vertex was added to new frontier
        // (Implementation would use reduction to check)
        
        std::swap(d_frontier, d_new_frontier);
        current_level++;
    }
}
```

## 3.7 Cooperative Groups

Modern CUDA provides cooperative groups for flexible thread coordination beyond traditional block boundaries.

### Basic Cooperative Groups Usage

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Thread block-level cooperation
__global__ void block_cooperation_example(float *data, int n) {
    auto block = cg::this_thread_block();
    
    int tid = block.thread_rank();
    int bid = block.group_index().x;
    int idx = bid * block.size() + tid;
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
    
    block.sync(); // Equivalent to __syncthreads()
    
    // Block-wide reduction using cooperative groups
    for (int offset = block.size() / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            data[idx] += data[idx + offset];
        }
        block.sync();
    }
}

// Warp-level cooperation
__global__ void warp_cooperation_example(float *data, int n) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Warp-level reduction using shuffle
        for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
            value += warp.shfl_down(value, offset);
        }
        
        // First thread in warp writes result
        if (warp.thread_rank() == 0) {
            data[idx / 32] = value;
        }
    }
}
```

### Multi-GPU Cooperative Kernels

```cuda
// Cooperative kernel that can span multiple GPUs
__global__ void __launch_bounds__(256)
multi_gpu_cooperative_kernel(float *data, int n, int gpu_id, int num_gpus) {
    auto grid = cg::this_grid();
    
    // Each GPU processes its portion
    int elements_per_gpu = (n + num_gpus - 1) / num_gpus;
    int start_idx = gpu_id * elements_per_gpu;
    int end_idx = min(start_idx + elements_per_gpu, n);
    
    for (int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x; 
         idx < end_idx; 
         idx += blockDim.x * gridDim.x) {
        data[idx] = sqrtf(data[idx]);
    }
    
    // Synchronize across all GPUs
    grid.sync();
    
    // Additional processing that requires all GPUs to be in sync
    if (grid.thread_rank() == 0) {
        printf("All GPUs have completed their portion\n");
    }
}

void launch_multi_gpu_kernel(float **d_data, int n, int num_gpus) {
    // Launch cooperative kernel across multiple GPUs
    cudaLaunchParams launch_params[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        
        launch_params[i].func = (void*)multi_gpu_cooperative_kernel;
        launch_params[i].gridDim = dim3(256);
        launch_params[i].blockDim = dim3(256);
        launch_params[i].sharedMem = 0;
        launch_params[i].stream = 0;
        launch_params[i].args = (void**)&d_data[i];
    }
    
    cudaLaunchCooperativeKernelMultiDevice(launch_params, num_gpus);
}
```

## Summary

This module covered advanced GPU algorithms and parallel patterns:

- **Reduction Algorithms**: Efficient aggregation using warp-level primitives and multi-pass strategies
- **Scan Operations**: Prefix sum computations with work-efficient algorithms
- **Sorting Algorithms**: Bitonic sort and radix sort implementations optimized for GPU architectures
- **Stencil Patterns**: Convolution and neighboring computations with shared memory optimization
- **Graph Algorithms**: BFS and other graph traversal algorithms adapted for GPU execution models
- **Cooperative Groups**: Modern CUDA programming model for flexible thread coordination

**Key Takeaways:**
1. GPU algorithms require careful consideration of work vs. depth complexity
2. Memory access patterns often dominate performance more than computational complexity
3. Cooperative groups provide flexible alternatives to traditional block-based synchronization
4. Multi-pass algorithms are often necessary for large datasets
5. Platform-specific optimizations can provide significant performance benefits

**Next Steps:**
- Practice implementing these fundamental patterns
- Learn to identify which pattern applies to your specific problem
- Experiment with different optimization techniques
- Explore domain-specific applications of these patterns
- Study performance scaling characteristics on different GPU architectures