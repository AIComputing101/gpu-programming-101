# Module 8: Domain-Specific Applications - Comprehensive Guide

> Environment note: The examples and integrations in this module assume Docker images with CUDA 12.9.1 and ROCm 7.0 (complete) are used for consistent library/tool availability.

## Introduction

Domain-specific GPU applications represent the practical implementation of GPU computing principles in real-world scenarios. This module explores how GPU acceleration transforms computational workflows across diverse fields including deep learning, scientific computing, image processing, computational finance, and data analytics.

Each domain presents unique computational patterns, performance requirements, and optimization opportunities. Understanding these domain-specific characteristics is crucial for developing efficient, scalable GPU applications that deliver significant performance improvements over traditional CPU-based implementations.

## Theoretical Foundations

### Domain-Specific Computing Principles

**Computational Characteristics:**
- **Arithmetic Intensity**: Ratio of computation to memory operations
- **Data Parallelism**: Degree of independent parallel operations
- **Memory Access Patterns**: Regular vs. irregular data access
- **Precision Requirements**: Single, double, half, or mixed precision

**Performance Optimization Framework:**
1. **Algorithm Analysis**: Identify computational bottlenecks
2. **Architecture Mapping**: Match algorithms to GPU capabilities
3. **Memory Optimization**: Minimize data movement costs
4. **Compute Optimization**: Maximize computational throughput

### GPU Architecture Considerations by Domain

**Deep Learning:**
- Tensor operations with high arithmetic intensity
- Regular memory access patterns suitable for coalescing
- Mixed precision opportunities with Tensor Cores
- Batch processing for improved parallelism

**Scientific Computing:**
- Numerical precision requirements (often double precision)
- Complex memory access patterns in PDE solvers
- Communication-intensive multi-GPU algorithms
- Irregular workloads in particle simulations

**Image Processing:**
- Spatial locality in 2D convolution operations
- Streaming processing for video applications
- Memory bandwidth critical for filter operations
- Real-time processing constraints

## 1. Deep Learning Applications

### Neural Network Fundamentals

**Core Operations:**
1. **Matrix Multiplication (GEMM)**: Foundation of dense layers
2. **Convolution**: Spatial feature extraction
3. **Activation Functions**: Non-linear transformations
4. **Normalization**: Batch/layer normalization operations

### Convolution Layer Implementation

**Direct Convolution Algorithm:**
```cpp
__global__ void conv2d_direct(const float* input, const float* weights,
                             float* output, int N, int C, int H, int W,
                             int K, int R, int S, int pad_h, int pad_w) {
    int n = blockIdx.x;
    int k = blockIdx.y;
    int h = blockIdx.z * blockDim.y + threadIdx.y;
    int w = threadIdx.x;
    
    if (h < H && w < W) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                    int h_in = h + r - pad_h;
                    int w_in = w + s - pad_w;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = n*C*H*W + c*H*W + h_in*W + w_in;
                        int weight_idx = k*C*R*S + c*R*S + r*S + s;
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        output[n*K*H*W + k*H*W + h*W + w] = sum;
    }
}
```

**Optimized Tiled Convolution:**
```cpp
__global__ void conv2d_tiled(const float* input, const float* weights,
                            float* output, int N, int C, int H, int W,
                            int K, int R, int S) {
    extern __shared__ float shared_mem[];
    float* tile_input = shared_mem;
    float* tile_weights = &shared_mem[TILE_SIZE * TILE_SIZE];
    
    // Cooperative loading of input and weight tiles
    // Compute convolution using shared memory
    // Minimize global memory accesses
}
```

### GEMM Optimization

**Basic GEMM Implementation:**
- Matrix tiling for shared memory usage
- Register blocking for reduced memory traffic
- Thread coarsening for improved arithmetic intensity

**Tensor Core Acceleration:**
```cpp
// NVIDIA Tensor Core utilization for mixed precision
__global__ void gemm_tensor_core(const half* A, const half* B, float* C,
                                int M, int N, int K) {
    // Use wmma API for Tensor Core operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::load_matrix_sync(a_frag, A + tile_offset, K);
    wmma::load_matrix_sync(b_frag, B + tile_offset, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C + tile_offset, c_frag, N, wmma::mem_row_major);
}
```

### Activation Functions

**ReLU Implementation:**
```cpp
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

**Fused Activation Functions:**
```cpp
__global__ void conv_relu_fused(const float* input, const float* weights,
                               float* output, int N, int C, int H, int W, int K) {
    // Compute convolution and apply ReLU in single kernel
    // Reduces memory bandwidth requirements
    float conv_result = compute_convolution(input, weights, ...);
    output[output_idx] = fmaxf(0.0f, conv_result);
}
```

### Batch Normalization

**Forward Pass:**
```cpp
__global__ void batch_norm_forward(const float* input, const float* gamma,
                                  const float* beta, float* output,
                                  float* mean, float* variance,
                                  int N, int C, int H, int W, float eps) {
    int c = blockIdx.x;
    
    // Compute mean and variance across batch
    float sum = 0.0f, sum_sq = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int hw = threadIdx.x; hw < H*W; hw += blockDim.x) {
            float val = input[n*C*H*W + c*H*W + hw];
            sum += val;
            sum_sq += val * val;
        }
    }
    
    // Reduction across threads
    __shared__ float ssum[256], ssum_sq[256];
    // ... reduction implementation
    
    // Normalize and scale
    float m = mean[c];
    float v = variance[c];
    float inv_std = rsqrtf(v + eps);
    
    for (int n = 0; n < N; n++) {
        for (int hw = threadIdx.x; hw < H*W; hw += blockDim.x) {
            int idx = n*C*H*W + c*H*W + hw;
            output[idx] = gamma[c] * (input[idx] - m) * inv_std + beta[c];
        }
    }
}
```

## 2. Scientific Computing Applications

### Computational Physics

**N-Body Simulation:**
```cpp
__global__ void nbody_kernel(float4* positions, float4* velocities,
                            float4* forces, int n, float dt, float eps) {
    extern __shared__ float4 shared_pos[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pos = positions[tid];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Tile-based force calculation
    for (int tile = 0; tile < gridDim.x; tile++) {
        // Load positions into shared memory
        shared_pos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
        __syncthreads();
        
        // Compute forces with particles in this tile
        for (int j = 0; j < blockDim.x; j++) {
            float4 other_pos = shared_pos[j];
            float dx = other_pos.x - pos.x;
            float dy = other_pos.y - pos.y;
            float dz = other_pos.z - pos.z;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float f = other_pos.w * inv_dist3;
            
            force.x += f * dx;
            force.y += f * dy;
            force.z += f * dz;
        }
        __syncthreads();
    }
    
    forces[tid] = force;
    
    // Update position and velocity
    float4 vel = velocities[tid];
    vel.x += force.x * dt / pos.w;
    vel.y += force.y * dt / pos.w;
    vel.z += force.z * dt / pos.w;
    
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    velocities[tid] = vel;
    positions[tid] = pos;
}
```

### Partial Differential Equations

**Heat Equation Solver:**
```cpp
__global__ void heat_equation_2d(float* u_new, const float* u_old,
                                 int nx, int ny, float alpha, float dt,
                                 float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < nx-1 && j < ny-1) {
        int idx = j * nx + i;
        
        float u_center = u_old[idx];
        float u_left = u_old[idx - 1];
        float u_right = u_old[idx + 1];
        float u_down = u_old[idx - nx];
        float u_up = u_old[idx + nx];
        
        float d2u_dx2 = (u_left - 2.0f*u_center + u_right) / (dx*dx);
        float d2u_dy2 = (u_down - 2.0f*u_center + u_up) / (dy*dy);
        
        u_new[idx] = u_center + alpha * dt * (d2u_dx2 + d2u_dy2);
    }
}
```

**Optimized Stencil Computation:**
```cpp
__global__ void stencil_3d_optimized(float* output, const float* input,
                                     int nx, int ny, int nz) {
    // Shared memory with halo regions
    __shared__ float s_data[BLOCK_Z+2][BLOCK_Y+2][BLOCK_X+2];
    
    // Load data with halo exchange
    // Compute stencil operation
    // Minimize global memory accesses through shared memory reuse
}
```

### Monte Carlo Methods

**Monte Carlo Pi Estimation:**
```cpp
__global__ void monte_carlo_pi(curandState* states, int* hits,
                              int samples_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[tid];
    
    int local_hits = 0;
    for (int i = 0; i < samples_per_thread; i++) {
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);
        if (x*x + y*y <= 1.0f) {
            local_hits++;
        }
    }
    
    hits[tid] = local_hits;
    states[tid] = local_state;
}
```

### Fast Fourier Transform

**Cooley-Tukey FFT Implementation:**
```cpp
__global__ void fft_cooley_tukey(float2* data, int n, int step, int stage) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n/2) {
        int k = tid % step;
        int j = 2 * step * (tid / step);
        
        float angle = -2.0f * M_PI * k / (2 * step);
        float2 twiddle = make_float2(cosf(angle), sinf(angle));
        
        float2 even = data[j + k];
        float2 odd = data[j + k + step];
        
        // Complex multiplication: odd * twiddle
        float2 temp;
        temp.x = odd.x * twiddle.x - odd.y * twiddle.y;
        temp.y = odd.x * twiddle.y + odd.y * twiddle.x;
        
        data[j + k] = make_float2(even.x + temp.x, even.y + temp.y);
        data[j + k + step] = make_float2(even.x - temp.x, even.y - temp.y);
    }
}
```

## 3. Image and Signal Processing

### Image Convolution

**2D Convolution with Shared Memory:**
```cpp
__global__ void image_convolution_2d(const float* image, const float* filter,
                                    float* output, int width, int height,
                                    int filter_size) {
    extern __shared__ float s_image[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int s_width = blockDim.x + filter_size - 1;
    int s_height = blockDim.y + filter_size - 1;
    
    // Load image data into shared memory with halo
    for (int dy = 0; dy < s_height; dy += blockDim.y) {
        for (int dx = 0; dx < s_width; dx += blockDim.x) {
            int sx = dx + tx;
            int sy = dy + ty;
            int gx = x + dx - tx;
            int gy = y + dy - ty;
            
            if (sx < s_width && sy < s_height) {
                if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                    s_image[sy * s_width + sx] = image[gy * width + gx];
                } else {
                    s_image[sy * s_width + sx] = 0.0f;
                }
            }
        }
    }
    __syncthreads();
    
    // Compute convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
                int sx = tx + fx;
                int sy = ty + fy;
                sum += s_image[sy * s_width + sx] * filter[fy * filter_size + fx];
            }
        }
        
        output[y * width + x] = sum;
    }
}
```

### Edge Detection

**Sobel Edge Detection:**
```cpp
__global__ void sobel_edge_detection(const float* input, float* output,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
        float gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                   -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                   -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
        
        // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
        float gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                   +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
        
        output[y*width + x] = sqrtf(gx*gx + gy*gy);
    }
}
```

## 4. Computational Finance

### Monte Carlo Option Pricing

**Black-Scholes Monte Carlo:**
```cpp
__global__ void monte_carlo_option_pricing(curandState* states,
                                          float* payoffs, float S0, float K,
                                          float r, float sigma, float T,
                                          int num_paths, int num_steps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;
    
    curandState local_state = states[tid];
    float dt = T / num_steps;
    float drift = (r - 0.5f * sigma * sigma) * dt;
    float diffusion = sigma * sqrtf(dt);
    
    float S = S0;
    for (int step = 0; step < num_steps; step++) {
        float z = curand_normal(&local_state);
        S *= expf(drift + diffusion * z);
    }
    
    // European call option payoff
    payoffs[tid] = fmaxf(S - K, 0.0f);
    states[tid] = local_state;
}
```

### Risk Calculations

**Value at Risk (VaR) Computation:**
```cpp
__global__ void var_calculation(const float* returns, float* sorted_returns,
                               int num_assets, int num_scenarios,
                               float confidence_level) {
    // Parallel sorting of scenario returns
    // Percentile calculation for VaR
    // Portfolio risk aggregation
}
```

## Performance Optimization Strategies

### Memory Access Optimization

**Coalesced Memory Access:**
- Ensure consecutive threads access consecutive memory locations
- Use appropriate data layouts (AoS vs SoA)
- Minimize stride patterns in memory access

**Shared Memory Utilization:**
- Cache frequently accessed data in shared memory
- Avoid bank conflicts through proper addressing
- Use shared memory for inter-thread communication

### Compute Optimization

**Arithmetic Intensity Improvement:**
- Increase computation per memory access
- Use fused multiply-add (FMA) operations
- Leverage built-in mathematical functions

**Thread Divergence Minimization:**
- Avoid conditional branches within warps
- Use predicated execution where possible
- Restructure algorithms to reduce divergence

### Mixed Precision Computing

**Half Precision Benefits:**
- Doubled memory bandwidth utilization
- Tensor Core acceleration on modern GPUs
- Reduced memory footprint

**Implementation Considerations:**
- Maintain numerical stability
- Use appropriate precision for different operations
- Implement loss scaling for training applications

## Cross-Platform Considerations

### CUDA-Specific Optimizations

**Warp-Level Primitives:**
```cpp
// Warp shuffle for efficient reductions
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Tensor Core Utilization:**
- Mixed precision training and inference
- WMMA API for matrix operations
- Optimal tile sizes for Tensor Core operations

### HIP/AMD Optimizations

**Wavefront-Aware Programming:**
```cpp
// AMD wavefront size is 64 threads
__global__ void amd_optimized_kernel() {
    int lane_id = threadIdx.x % 64;
    // Wavefront-specific optimizations
}
```

**LDS (Local Data Share) Optimization:**
- Bank conflict avoidance (32 banks)
- Optimal data layouts for AMD architecture
- Efficient inter-wavefront communication

## Conclusion

Domain-specific GPU applications require deep understanding of both the target domain and GPU architecture. Success factors include:

1. **Domain Expertise**: Understanding computational requirements and constraints
2. **Algorithm Selection**: Choosing GPU-friendly algorithms and data structures
3. **Architecture Awareness**: Optimizing for specific GPU capabilities
4. **Performance Analysis**: Continuous profiling and optimization
5. **Cross-Platform Design**: Ensuring portability across different GPU vendors

The applications covered in this module demonstrate how GPU acceleration can transform computational workflows across diverse domains, often providing order-of-magnitude performance improvements while enabling new computational possibilities.