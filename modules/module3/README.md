# Module 3: GPU Architecture and Execution Models

This module covers advanced GPU programming concepts focusing on fundamental parallel algorithms and optimization patterns essential for high-performance computing.

## Learning Objectives

- Master fundamental parallel algorithm patterns (reduction, scan, sorting)
- Understand algorithm complexity trade-offs (work vs depth complexity)
- Implement efficient stencil and convolution computations
- Apply modern CUDA cooperative groups programming model
- Design scalable algorithms for large datasets
- Analyze and optimize parallel algorithm performance

## Prerequisites

- Completion of Module 1 (GPU Programming Foundations)
- Completion of Module 2 (Advanced Memory Management)
- Understanding of basic parallel computing concepts
- Familiarity with algorithm complexity analysis

## Contents

### Core Content
- **content.md** - Comprehensive guide covering all advanced algorithm patterns

### Examples

#### 1. Reduction Algorithms (`01_reduction_algorithms_*.cu/.cpp`)
- Naive parallel reduction implementation
- Optimized reduction with shared memory
- Modern warp-level reductions using cooperative groups
- Multi-pass reduction for large datasets
- Performance comparison and analysis

**Key Concepts:**
- Thread divergence optimization
- Bank conflict avoidance
- Warp-level primitives
- Work complexity vs depth complexity

#### 2. Scan (Prefix Sum) Algorithms (`02_scan_prefix_sum_*.cu/.cpp`)
- Naive inclusive/exclusive scan
- Hillis-Steele scan (O(n log n) work, O(log n) depth)
- Blelloch scan (O(n) work, O(log n) depth)
- Segmented scan for multiple sequences
- Large array scanning strategies
- AMD wavefront-optimized implementations

**Key Concepts:**
- Work-efficient vs step-efficient algorithms
- Up-sweep and down-sweep phases
- Multi-level scanning approaches
- Wavefront vs warp optimizations

#### 3. Sorting Algorithms (`03_sorting_algorithms_*.cu/.cpp`)
- Bitonic sorting network
- Parallel comparison-based sorting
- Radix sort implementation
- Odd-even sort for small arrays
- Performance comparison with CPU algorithms

**Key Concepts:**
- Sorting networks
- Data-parallel sorting strategies
- Algorithm scalability analysis
- Cross-platform optimizations

#### 4. Convolution and Stencil (`04_convolution_stencil_*.cu/.cpp`)
- 1D stencil computations with shared memory
- 2D convolution with halo region handling
- 3D stencil operations
- Separable convolution optimization
- Boundary condition management

**Key Concepts:**
- Halo region loading
- Memory coalescing in stencil patterns
- Shared memory tiling strategies
- Separable filters for performance

#### 5. Matrix Operations (`05_matrix_operations_cuda.cu`)
- Tiled matrix multiplication with shared memory
- Matrix transpose with bank conflict avoidance
- Matrix-vector multiplication using cooperative groups
- Performance analysis and GFLOPS calculation

**Key Concepts:**
- Shared memory tiling patterns
- Memory bandwidth optimization
- Cooperative groups programming
- Performance metrics calculation

#### 6. Graph Algorithms (`06_graph_algorithms_cuda.cu`)
- Breadth-First Search (BFS) with frontier queues
- Single-Source Shortest Path (SSSP)
- Connected Components labeling
- Triangle counting
- PageRank power iteration

**Key Concepts:**
- CSR graph representation
- Frontier-based graph traversal
- Atomic operations for graph updates
- Load balancing in irregular computations

#### 7. Cooperative Groups (`07_cooperative_groups_cuda.cu`)
- Modern CUDA cooperative groups API
- Warp-level primitives and shuffle operations
- Block-level and grid-level coordination
- Advanced synchronization patterns
- Multi-GPU cooperative kernels

**Key Concepts:**
- Cooperative groups hierarchy
- Warp-level programming
- Modern CUDA synchronization
- Multi-device coordination

### Cross-Platform Support
- **HIP Examples**: Cross-platform implementations supporting both AMD and NVIDIA GPUs
- Platform-specific optimizations for wavefront (64 threads) vs warp (32 threads) architectures
- Complete HIP implementations for reduction, scan, sorting, and convolution algorithms

## Quick Start

### Building Examples

```bash
# Build all CUDA examples
make cuda

# Build all HIP examples (requires HIP compiler)
make hip

# Build specific example
make 01_reduction_algorithms_cuda
```

### Running Tests

```bash
# Run all tests
make test

# Run performance benchmarks
make test_performance

# Test compilation only
make test_compile
```

### Individual Examples

```bash
# Reduction algorithms
make reduction
./01_reduction_algorithms_cuda

# Scan algorithms
make scan
./02_scan_prefix_sum_cuda

# Sorting algorithms
make sorting
./03_sorting_algorithms_cuda

# Convolution/Stencil
make convolution
./04_convolution_stencil_cuda

# Matrix operations
make matrix
./05_matrix_operations_cuda

# Graph algorithms
make graph
./06_graph_algorithms_cuda

# Cooperative groups
make cooperative
./07_cooperative_groups_cuda

# HIP examples (cross-platform)
make scan_hip
./02_scan_prefix_sum_hip
```

## Algorithm Performance Analysis

### Complexity Comparison

| Algorithm | Naive | Optimized | Work Complexity | Depth Complexity |
|-----------|-------|-----------|-----------------|-------------------|
| Reduction | O(n²) | O(n) | O(n) | O(log n) |
| Scan (Hillis-Steele) | O(n²) | O(n log n) | O(n log n) | O(log n) |
| Scan (Blelloch) | - | O(n) | O(n) | O(log n) |
| Bitonic Sort | - | O(n log² n) | O(n log² n) | O(log² n) |
| Matrix Multiply | O(n³) | O(n³) | O(n³) | O(log n) |
| BFS | O(V + E) | O(V + E) | O(V + E) | O(D) |
| Triangle Count | O(V³) | O(E^1.5) | O(E^1.5) | O(log V) |

### Performance Benchmarking

Use the provided benchmarking targets:

```bash
# Run performance analysis
make test_performance

# Show profiling commands
make profile_algorithms

# Algorithm complexity analysis
make analyze_complexity
```

## Key Programming Patterns

### 1. Cooperative Groups (Modern CUDA)

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void modernReduction(float *data) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += warp.shfl_down(value, offset);
    }
}
```

### 2. Shared Memory Optimization

```cuda
__global__ void optimizedStencil(float *input, float *output) {
    __shared__ float shared[BLOCK_SIZE + 2*RADIUS];
    
    // Load data + halo regions
    // Apply computation using shared memory
    // Minimize global memory accesses
}
```

### 3. Multi-Pass Algorithms

```cuda
// Handle large datasets with multiple kernel launches
void largeArrayReduction(float *data, int n) {
    while (n > 1) {
        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        reductionKernel<<<blocks, THREADS_PER_BLOCK>>>(data, n);
        n = blocks;
    }
}
```

## Profiling and Optimization

### NVIDIA Profiling Commands

```bash
# Throughput analysis
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./01_reduction_algorithms_cuda

# Memory analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./04_convolution_stencil_cuda

# Shared memory bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./02_scan_prefix_sum_cuda
```

## Advanced Topics Covered

### Algorithm Design Principles
- **Work Complexity**: Total operations performed
- **Depth Complexity**: Critical path length
- **Memory Complexity**: Space requirements
- **Communication Complexity**: Data movement costs

### Optimization Strategies
- **Memory Access Patterns**: Coalescing and caching
- **Thread Divergence**: Minimizing warp divergence
- **Resource Utilization**: Occupancy optimization
- **Algorithm Selection**: Choosing appropriate algorithms for problem size

### Scalability Considerations
- **Large Dataset Handling**: Multi-pass algorithms
- **Multi-GPU Scaling**: Algorithm decomposition
- **Memory Hierarchy**: Utilizing all levels of GPU memory

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Ensure CUDA/HIP toolkit is properly installed
   - Check compute capability requirements
   - Verify C++11 support

2. **Runtime Errors**
   - Check array bounds and memory allocation
   - Verify kernel launch parameters
   - Use CUDA_CHECK macro for error handling

3. **Performance Issues**
   - Profile memory access patterns
   - Check for thread divergence
   - Analyze occupancy and resource usage

### Hardware Requirements

- **CUDA Examples**: NVIDIA GPU with compute capability 5.0+
- **HIP Examples**: AMD GPU (GCN architecture) or NVIDIA GPU
- **Memory**: Examples scale with available GPU memory

## Further Reading

- CUDA Programming Guide (Algorithm Patterns section)
- "Programming Massively Parallel Processors" by Kirk & Hwu
- NVIDIA Performance Tuning Guide
- AMD ROCm Documentation for HIP programming

## Exercise Suggestions

1. **Implement Custom Algorithms**: Create your own reduction or scan variants
2. **Performance Optimization**: Optimize existing examples for your specific hardware
3. **Algorithm Comparison**: Benchmark different algorithmic approaches
4. **Multi-GPU Scaling**: Extend examples to use multiple GPUs
5. **Real-World Applications**: Apply patterns to specific problem domains

---

**Note**: This module provides both educational implementations (showing algorithm progression) and optimized versions. Focus on understanding the concepts before optimizing for specific use cases.