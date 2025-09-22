# Module 6: Fundamental Parallel Algorithms - Comprehensive Guide

> Environment note: The examples and benchmarks in this module are tested in Docker with CUDA 13.0.1 (Ubuntu 24.04) and ROCm 7.0.1 (Ubuntu 24.04) to ensure reproducibility. Recent algorithm fixes improve performance.

## Introduction

Fundamental parallel algorithms form the core building blocks of high-performance GPU computing. These algorithms—convolution, stencil computations, histogram operations, reduction patterns, and prefix sum operations—appear across virtually all domains of parallel computing, from scientific simulation to machine learning, image processing to data analytics.

This module provides comprehensive coverage of these essential algorithms, emphasizing both theoretical understanding and practical, optimized implementations. Each algorithm is presented with multiple variants, from simple educational implementations to production-optimized versions that achieve near-peak performance on modern GPU architectures.

## Theoretical Foundations

### Parallel Algorithm Design Principles

**Work and Depth Complexity:**
- **Work (W)**: Total number of operations performed
- **Depth (D)**: Length of the longest dependency chain
- **Parallelism**: P = W/D represents the maximum theoretical speedup

**Key Design Considerations:**
1. **Data Locality**: Minimize memory traffic through cache-friendly access patterns
2. **Load Balance**: Distribute work evenly across processing elements
3. **Communication**: Minimize inter-thread communication and synchronization
4. **Scalability**: Ensure algorithms scale efficiently with problem size and core count

**GPU-Specific Considerations:**
- **Memory Hierarchy**: Leverage shared memory, registers, and cache effectively
- **Thread Divergence**: Minimize branching within warps/wavefronts
- **Occupancy**: Balance resource usage to maximize concurrent threads
- **Memory Coalescing**: Ensure efficient memory access patterns

## 1. Convolution Algorithms

### Mathematical Foundation

Convolution is a fundamental operation in signal processing and computer vision:

**1D Discrete Convolution:**
```
(f * g)[n] = Σ f[m] * g[n - m]
             m
```

**2D Discrete Convolution:**
```
(f * g)[x,y] = ΣΣ f[i,j] * g[x - i, y - j]
               i j
```

### Algorithm Variants

#### Direct Convolution
**Complexity:** O(N * M) for 1D, O(N² * M²) for 2D
- Simple implementation with nested loops
- Good for small kernels (M < 16)
- Memory-bound for large inputs

#### Separable Convolution
**Complexity:** O(N² * M) for 2D separable kernels
- Decomposes 2D kernel into 1D operations
- Reduces complexity from O(N² * M²) to O(N² * M)
- Applicable when kernel is separable: K(x,y) = K₁(x) * K₂(y)

#### FFT-based Convolution
**Complexity:** O(N log N)
- Uses Fast Fourier Transform: F⁻¹(F(f) * F(g))
- Efficient for large kernels (M > 64)
- Requires complex arithmetic and additional memory

### Implementation Strategies

**Memory Access Optimization:**
1. **Shared Memory Tiling**: Load data blocks into shared memory
2. **Halo Region Loading**: Handle boundary conditions efficiently
3. **Coalesced Access**: Ensure consecutive threads access consecutive memory
4. **Register Reuse**: Maximize data reuse within thread registers

**Boundary Condition Handling:**
- **Zero Padding**: Extend input with zeros
- **Reflection**: Mirror boundary values
- **Periodic**: Wrap around boundaries
- **Constant**: Use fixed boundary value

### Performance Analysis

**Memory Bandwidth Requirements:**
- Input bandwidth: N bytes/operation
- Output bandwidth: N bytes/operation  
- Kernel bandwidth: M bytes (reused across operations)
- Total: 2N + M bytes per output element

**Arithmetic Intensity:**
- Operations per output: M multiply-add operations
- Bytes per output: 2N + M (assuming no reuse)
- AI = M / (2N + M) multiply-adds per byte

**Optimization Goals:**
- Maximize shared memory utilization
- Minimize global memory transactions
- Achieve high arithmetic intensity through data reuse

## 2. Stencil Computations

### Mathematical Foundation

Stencil operations apply a fixed pattern of computation to each point in a grid:

**General Form:**
```
u'[i] = Σ c[j] * u[i + j]
        j∈S
```

Where S is the stencil pattern and c[j] are coefficients.

**Common Stencil Patterns:**
- **3-point 1D**: [-1, 0, 1] (first derivative approximation)
- **5-point 2D**: Cross pattern for Laplacian operator
- **7-point 3D**: Six neighbors plus center point
- **27-point 3D**: Full neighborhood stencil

### Algorithm Design

#### Memory Access Patterns
**Ghost Cell Management:**
- Extend computational domain with boundary values
- Load ghost cells into shared memory for efficient access
- Handle different boundary condition types

**Temporal Blocking:**
- Compute multiple time steps within shared memory
- Reduces global memory traffic
- Requires careful dependency analysis

#### Optimization Techniques

**Thread Coarsening:**
```cuda
// Each thread processes multiple points
for (int i = 0; i < COARSENING_FACTOR; i++) {
    int idx = blockIdx.x * blockDim.x * COARSENING_FACTOR + 
              threadIdx.x + i * blockDim.x;
    if (idx < n) {
        output[idx] = stencil_operation(input, idx);
    }
}
```

**Register Blocking:**
- Keep frequently accessed values in registers
- Shift register values for sliding window operations
- Reduces shared memory bank conflicts

### Performance Considerations

**Memory Bandwidth:**
- Each point requires S memory accesses
- Output requires 1 write operation
- Ghost cells reduce effective bandwidth utilization

**Arithmetic Intensity:**
- S multiply-add operations per point
- (S + 1) memory operations per point
- AI = S / (S + 1) operations per byte

**Scalability:**
- Performance scales with problem size until memory bandwidth limit
- Multi-GPU scaling requires inter-GPU communication for ghost cells
- Communication-computation overlap critical for large problems

## 3. Histogram Operations

### Problem Formulation

Histogram construction counts occurrences of values within specified bins:

**Basic Histogram:**
```
for each input value v:
    bin = compute_bin(v)
    histogram[bin]++
```

**Challenges in Parallel Implementation:**
- Race conditions in bin updates
- Atomic operation contention
- Load imbalance across bins
- Memory access patterns

### Atomic Operations Approach

**Direct Atomic Increment:**
```cuda
__global__ void atomic_histogram(int *input, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = compute_bin(input[idx]);
        atomicAdd(&histogram[bin], 1);
    }
}
```

**Performance Characteristics:**
- Simple implementation
- High contention for popular bins
- Performance degrades with skewed distributions
- Memory bandwidth underutilized due to atomic serialization

### Privatization Strategy

**Per-Block Private Histograms:**
```cuda
__global__ void privatized_histogram(int *input, int *histogram, int n) {
    extern __shared__ int private_hist[];
    
    // Initialize private histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        private_hist[i] = 0;
    }
    __syncthreads();
    
    // Build private histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = compute_bin(input[idx]);
        atomicAdd(&private_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (private_hist[i] > 0) {
            atomicAdd(&histogram[i], private_hist[i]);
        }
    }
}
```

**Advantages:**
- Reduces global memory contention
- Better performance for uniform distributions
- Scales with number of blocks

**Disadvantages:**
- Requires additional memory (NUM_BINS per block)
- Final merge step still requires global atomics
- May not fit in shared memory for large bin counts

### Coarsening Techniques

**Thread Coarsening:**
- Each thread processes multiple input elements
- Builds local count before atomic update
- Reduces atomic operation frequency

**Warp-Level Aggregation:**
- Use warp shuffle operations to aggregate within warp
- Single atomic operation per warp per bin
- Significant reduction in atomic contention

## 4. Reduction Patterns

### Algorithm Variants

#### Tree Reduction (Hillis-Steele)

**Characteristics:**
- Work: O(n log n)
- Depth: O(log n)  
- Simple implementation
- Not work-efficient for large inputs

**Implementation Pattern:**
```cuda
__global__ void tree_reduction(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    // Load data to shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

#### Optimized Tree Reduction (Brent-Kung)

**Characteristics:**
- Work: O(n)
- Depth: O(log n)
- Work-efficient implementation
- More complex but better scaling

**Key Optimizations:**
1. **Avoid Thread Divergence**: Use contiguous thread access
2. **Bank Conflict Avoidance**: Careful shared memory indexing
3. **Loop Unrolling**: Reduce loop overhead for small strides

#### Modern Warp-Level Reduction

**Using Cooperative Groups:**
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ float warp_reduce(float val) {
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Warp-level reduction using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        val += tile32.shfl_down(val, offset);
    }
    return val;
}
```

### Multi-level Reduction

**Large Array Handling:**
1. **Block-level Reduction**: Reduce within each block
2. **Inter-block Reduction**: Reduce block results
3. **Recursive Application**: For very large arrays

**Kernel Launch Strategy:**
```cuda
void multi_level_reduction(float *input, float *output, int n) {
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float *temp_output;
    
    while (n > 1) {
        cudaMalloc(&temp_output, blocks * sizeof(float));
        reduction_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, temp_output, n);
        
        input = temp_output;
        n = blocks;
        blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    }
    
    cudaMemcpy(output, input, sizeof(float), cudaMemcpyDeviceToHost);
}
```

## 5. Prefix Sum Algorithms

### Algorithm Theory

**Prefix Sum Definition:**
Given array [a₀, a₁, ..., aₙ₋₁], compute:
- **Inclusive**: [a₀, a₀+a₁, a₀+a₁+a₂, ...]
- **Exclusive**: [0, a₀, a₀+a₁, a₀+a₁+a₂, ...]

### Hillis-Steele Algorithm

**Characteristics:**
- Work: O(n log n)
- Depth: O(log n)
- Step-efficient (minimum depth)
- High memory bandwidth requirements

**Implementation:**
```cuda
__global__ void hillis_steele_scan(float *input, float *output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input data
    temp[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            temp[tid] += val;
        }
        __syncthreads();
    }
    
    // Write output
    if (idx < n) {
        output[idx] = temp[tid];
    }
}
```

### Blelloch Algorithm (Work-Efficient)

**Characteristics:**
- Work: O(n)
- Depth: O(log n)
- Work-efficient
- Two-phase algorithm (up-sweep + down-sweep)

**Up-sweep Phase:**
```cuda
// Build tree bottom-up
for (int stride = 1; stride < n; stride *= 2) {
    for (int i = stride - 1; i < n; i += 2 * stride) {
        temp[i + stride] += temp[i];
    }
}
```

**Down-sweep Phase:**
```cuda
// Set root to identity
temp[n - 1] = 0;

// Propagate down
for (int stride = n / 2; stride > 0; stride /= 2) {
    for (int i = stride - 1; i < n; i += 2 * stride) {
        float t = temp[i];
        temp[i] = temp[i + stride];
        temp[i + stride] += t;
    }
}
```

### Segmented Scan

**Problem**: Perform independent scans on multiple segments within array

**Approach**: Use flag array to identify segment boundaries
```cuda
if (flags[i] == 1) {
    // Start of new segment - don't propagate from previous
    result[i] = input[i];
} else {
    // Continue current segment
    result[i] = input[i] + result[i-1];
}
```

**Applications:**
- Sparse matrix operations
- Variable-length sequence processing
- Parallel parsing algorithms

### Large Array Scanning

**Multi-phase Approach:**
1. **Phase 1**: Scan within blocks, save block sums
2. **Phase 2**: Scan block sums to get per-block offsets  
3. **Phase 3**: Add block offsets to within-block results

**Memory Access Optimization:**
- Minimize global memory transactions
- Maximize shared memory utilization
- Overlap computation with memory access

## Cross-Platform Optimization Strategies

### NVIDIA CUDA Optimizations

**Warp-Level Primitives:**
- `__shfl_down()` for intra-warp communication
- `__ballot_sync()` for warp voting operations
- `__activemask()` for active thread detection

**Cooperative Groups:**
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

auto block = cg::this_thread_block();
auto warp = cg::tiled_partition<32>(block);
```

**Memory Hierarchy:**
- L1 cache configuration (prefer shared vs. L1)
- Texture memory for read-only data
- Constant memory for small, uniform data

### AMD ROCm Optimizations

**Wavefront-Level Operations:**
- 64-thread wavefronts vs. 32-thread warps
- Different shuffle patterns and voting operations
- LDS (Local Data Store) optimization

**Memory Subsystem:**
- Different cache hierarchy
- Memory channel interleaving
- Infinity Fabric for multi-GPU

**Platform-Specific Intrinsics:**
```cpp
// AMD-specific wavefront operations
#ifdef __HIP_PLATFORM_AMD__
    result = __shfl_down(value, offset, 64);  // 64-thread wavefront
#else
    result = __shfl_down_sync(0xffffffff, value, offset, 32);  // 32-thread warp
#endif
```

### HIP Portability

**Single Source Code:**
```cpp
#include <hip/hip_runtime.h>

// HIP kernel works on both NVIDIA and AMD
__global__ void portable_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Portable implementation
        data[idx] = process_data(data[idx]);
    }
}
```

**Runtime Architecture Detection:**
```cpp
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);

if (prop.major >= 7) {  // Turing or later
    // Use Tensor Cores if available
} else {
    // Fallback implementation
}
```

## Performance Analysis and Optimization

### Profiling Methodology

**NVIDIA Profiling:**
```bash
# Compute throughput analysis
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./algorithm

# Memory bandwidth analysis  
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./algorithm

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./algorithm
```

**AMD Profiling:**
```bash
# Basic performance metrics
rocprof --stats ./algorithm

# Detailed memory analysis
rocprof --hip-trace --stats ./algorithm

# Hardware performance counters
rocprof --hsa-trace --stats ./algorithm
```

### Optimization Guidelines

**Memory Optimization:**
1. **Coalescing**: Ensure consecutive threads access consecutive memory
2. **Bank Conflicts**: Avoid shared memory bank conflicts through padding
3. **Cache Utilization**: Design access patterns for optimal cache usage
4. **Bandwidth Utilization**: Achieve high percentage of peak bandwidth

**Compute Optimization:**
1. **Occupancy**: Balance register usage, shared memory, and thread count
2. **Divergence**: Minimize branching within warps/wavefronts  
3. **Instruction Mix**: Balance arithmetic operations with memory operations
4. **Loop Optimization**: Unroll loops to reduce overhead

**Algorithm-Specific Optimization:**
1. **Data Reuse**: Maximize temporal and spatial locality
2. **Work Distribution**: Balance load across processing elements
3. **Communication**: Minimize inter-thread synchronization
4. **Scalability**: Design for multiple GPU scaling

### Performance Expectations

**Typical Performance Gains:**
- **Convolution**: 10-100x over CPU, depending on kernel size
- **Stencil**: 5-50x over CPU, limited by memory bandwidth
- **Histogram**: 2-20x over CPU, varies with distribution
- **Reduction**: 50-500x over CPU for large arrays
- **Prefix Sum**: 10-100x over CPU, algorithm-dependent

**Scalability Characteristics:**
- **Problem Size**: Performance increases with array size until memory limit
- **Multi-GPU**: Near-linear scaling for embarrassingly parallel algorithms
- **Architecture**: Performance scales with memory bandwidth and compute units

## Applications and Use Cases

### Scientific Computing

**Computational Fluid Dynamics:**
- Stencil operations for finite difference methods
- Reduction operations for convergence checking
- Prefix sum for particle sorting and load balancing

**Image Processing:**
- Convolution for filters and feature detection
- Histogram for intensity analysis and equalization
- Reduction for global statistics computation

**Molecular Dynamics:**
- Reduction for energy and force calculations
- Prefix sum for particle neighbor list construction
- Histogram for radial distribution functions

### Machine Learning

**Convolutional Neural Networks:**
- Optimized convolution for forward and backward passes
- Reduction for batch normalization and loss computation
- Prefix sum for attention mechanisms

**Data Preprocessing:**
- Histogram for data distribution analysis
- Prefix sum for cumulative probability computation
- Reduction for normalization statistics

**Training Optimization:**
- All-reduce for gradient aggregation in distributed training
- Scan operations for dynamic batching
- Histogram for gradient distribution analysis

### Data Analytics

**Database Operations:**
- Prefix sum for parallel select and join operations
- Reduction for aggregation queries (SUM, COUNT, AVG)
- Histogram for data profiling and statistics

**Stream Processing:**
- Sliding window reductions for real-time analytics
- Prefix sum for cumulative metrics computation
- Histogram for anomaly detection

**Graph Analytics:**
- Reduction for vertex property aggregation
- Prefix sum for edge list processing
- Histogram for degree distribution analysis

## Summary

Module 6 provides comprehensive coverage of fundamental parallel algorithms that form the building blocks of high-performance GPU computing:

**Key Algorithmic Concepts:**
- **Work vs. Depth Complexity**: Understanding theoretical performance limits
- **Memory Access Patterns**: Designing cache-friendly algorithms
- **Load Balancing**: Distributing work efficiently across GPU cores
- **Cross-Platform Optimization**: Creating portable high-performance code

**Practical Skills Developed:**
- **Algorithm Implementation**: From naive to highly optimized versions
- **Performance Analysis**: Using profiling tools to identify bottlenecks
- **Optimization Techniques**: Applying GPU-specific optimization strategies
- **Cross-Platform Development**: Creating code that works efficiently on multiple architectures

**Real-World Applications:**
- **Scientific Simulation**: CFD, molecular dynamics, image processing
- **Machine Learning**: CNN operations, data preprocessing, training optimization
- **Data Analytics**: Database operations, stream processing, graph analytics

These fundamental algorithms appear repeatedly in more complex applications and form the foundation for the advanced topics covered in subsequent modules. Mastering these patterns is essential for developing efficient GPU-accelerated software across all domains of parallel computing.

The combination of theoretical understanding, practical implementation experience, and cross-platform optimization skills developed in this module provides the foundation for tackling increasingly complex parallel computing challenges in scientific, industrial, and research applications.