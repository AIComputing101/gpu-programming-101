#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

#define MATRIX_SIZE 2048
#define TILE_SIZE 32
#define BLOCK_SIZE 256
#define VECTOR_SIZE (32 * 1024 * 1024)  // 32M elements
#define NUM_ITERATIONS 5

// Constant memory for lookup tables
__constant__ float const_lookup_table[256];

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timer class for performance measurement
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float endTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// 1. Memory Coalescing Examples
//=================================

// Poor coalescing: strided memory access
__global__ void stridedMemoryAccess(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    
    if (strided_idx < n) {
        // Poor coalescing - threads access memory with large stride
        data[strided_idx] = data[strided_idx] * 2.0f + 1.0f;
    }
}

// Good coalescing: sequential memory access
__global__ void coalescedMemoryAccess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Good coalescing - threads access contiguous memory
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Improved coalescing with vectorized loads
__global__ void vectorizedMemoryAccess(float4 *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Vectorized load - 16 bytes per transaction
        float4 vec = data[idx];
        vec.x = vec.x * 2.0f + 1.0f;
        vec.y = vec.y * 2.0f + 1.0f;
        vec.z = vec.z * 2.0f + 1.0f;
        vec.w = vec.w * 2.0f + 1.0f;
        data[idx] = vec;
    }
}

// 2. Shared Memory Optimization Examples
//======================================

// Matrix transpose with poor shared memory usage
__global__ void naiveTranspose(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Direct transpose - poor memory coalescing on write
        output[x * height + y] = input[y * width + x];
    }
}

// Matrix transpose with shared memory tiling
__global__ void tiledTranspose(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Coalesced read from input
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Coalesced write to output (transposed coordinates)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Bank conflict demonstration
__global__ void bankConflictKernel(float *input, float *output, int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Bank conflicts - multiple threads access same bank
    int conflict_idx = (tid * 33) % BLOCK_SIZE; // Creates bank conflicts
    float value = shared_data[conflict_idx];
    
    __syncthreads();
    
    if (idx < n) {
        output[idx] = value;
    }
}

// Bank conflict free version
__global__ void noBankConflictKernel(float *input, float *output, int n) {
    __shared__ float shared_data[BLOCK_SIZE + 32]; // Padding to avoid conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // No bank conflicts - structured access pattern
    float value = shared_data[tid];
    
    __syncthreads();
    
    if (idx < n) {
        output[idx] = value * 2.0f;
    }
}

// 3. Constant Memory Usage Examples
//=================================

// Using constant memory for lookup tables
__global__ void constantMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Use constant memory lookup table
        int lookup_idx = (int)(fabsf(value) * 255.0f) % 256;
        float lookup_value = const_lookup_table[lookup_idx];
        
        data[idx] = value * lookup_value;
    }
}

// Regular global memory version for comparison
__global__ void globalMemoryLookupKernel(float *data, float *lookup_table, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Use global memory lookup table - slower than constant memory
        int lookup_idx = (int)(fabsf(value) * 255.0f) % 256;
        float lookup_value = lookup_table[lookup_idx];
        
        data[idx] = value * lookup_value;
    }
}

// 4. Cache Optimization Examples
//==============================

// Cache-friendly matrix multiplication (tiled)
__global__ void tiledMatrixMul(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < width && t * TILE_SIZE + threadIdx.x < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < width && t * TILE_SIZE + threadIdx.y < width) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute using shared memory (cache-friendly)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// Cache-unfriendly matrix multiplication (naive)
__global__ void naiveMatrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Direct global memory access - poor cache utilization
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        
        C[row * width + col] = sum;
    }
}

// 5. Memory Layout Optimization
//=============================

// Structure of Arrays (SoA) - good for GPU
struct ParticlesSoA {
    float *x, *y, *z;        // Position arrays
    float *vx, *vy, *vz;     // Velocity arrays
    float *mass;             // Mass array
};

// Array of Structures (AoS) - poor for GPU coalescing
struct ParticleAoS {
    float x, y, z;           // Position
    float vx, vy, vz;        // Velocity  
    float mass;              // Mass
};

// Update particles using SoA layout (good coalescing)
__global__ void updateParticlesSoA(float *x, float *y, float *z,
                                   float *vx, float *vy, float *vz,
                                   float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Coalesced memory access - each thread accesses contiguous elements
        x[idx] += vx[idx] * dt;
        y[idx] += vy[idx] * dt;
        z[idx] += vz[idx] * dt;
    }
}

// Update particles using AoS layout (poor coalescing)
__global__ void updateParticlesAoS(ParticleAoS *particles, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Non-coalesced memory access - each thread accesses strided memory
        particles[idx].x += particles[idx].vx * dt;
        particles[idx].y += particles[idx].vy * dt;
        particles[idx].z += particles[idx].vz * dt;
    }
}

// Performance test functions
void testMemoryCoalescing() {
    printf("=== Memory Coalescing Performance Test ===\n");
    
    float *d_data;
    float4 *d_data_vec;
    size_t size = VECTOR_SIZE * sizeof(float);
    size_t size_vec = (VECTOR_SIZE / 4) * sizeof(float4);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_data_vec, size_vec));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((VECTOR_SIZE + block.x - 1) / block.x);
    dim3 grid_vec((VECTOR_SIZE / 4 + block.x - 1) / block.x);
    
    CudaTimer timer;
    
    // Test coalesced access
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        coalescedMemoryAccess<<<grid, block>>>(d_data, VECTOR_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float coalesced_time = total_time / NUM_ITERATIONS;
    
    // Test strided access (stride 2)
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        stridedMemoryAccess<<<dim3(grid.x/2), block>>>(d_data, VECTOR_SIZE, 2);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float strided_2_time = total_time / NUM_ITERATIONS;
    
    // Test strided access (stride 32)
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        stridedMemoryAccess<<<dim3(grid.x/32), block>>>(d_data, VECTOR_SIZE, 32);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float strided_32_time = total_time / NUM_ITERATIONS;
    
    // Test vectorized access
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        vectorizedMemoryAccess<<<grid_vec, block>>>(d_data_vec, VECTOR_SIZE/4);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float vectorized_time = total_time / NUM_ITERATIONS;
    
    // Calculate bandwidth
    float data_gb = VECTOR_SIZE * sizeof(float) / 1e9;
    
    printf("Coalesced access:      %.3f ms (%.1f GB/s)\n", 
           coalesced_time, 2 * data_gb / (coalesced_time / 1000.0));
    printf("Strided access (2):    %.3f ms (%.1f GB/s) [%.1fx slower]\n", 
           strided_2_time, data_gb / (strided_2_time / 1000.0), strided_2_time / coalesced_time);
    printf("Strided access (32):   %.3f ms (%.1f GB/s) [%.1fx slower]\n", 
           strided_32_time, data_gb / (strided_32_time / 1000.0), strided_32_time / coalesced_time);
    printf("Vectorized access:     %.3f ms (%.1f GB/s) [%.1fx %s]\n", 
           vectorized_time, 2 * data_gb / (vectorized_time / 1000.0),
           fabs(vectorized_time / coalesced_time), 
           vectorized_time < coalesced_time ? "faster" : "slower");
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_data_vec));
}

void testSharedMemoryOptimization() {
    printf("=== Shared Memory Optimization Test ===\n");
    
    float *d_input, *d_output;
    size_t matrix_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_input, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_output, matrix_size));
    
    // Initialize with test data
    std::vector<float> h_data(MATRIX_SIZE * MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_data[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_data.data(), matrix_size, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((MATRIX_SIZE + block.x - 1) / block.x, (MATRIX_SIZE + block.y - 1) / block.y);
    dim3 tile_grid((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    CudaTimer timer;
    
    // Test naive transpose
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        naiveTranspose<<<grid, block>>>(d_input, d_output, MATRIX_SIZE, MATRIX_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float naive_time = total_time / NUM_ITERATIONS;
    
    // Test tiled transpose
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        tiledTranspose<<<tile_grid, dim3(TILE_SIZE, TILE_SIZE)>>>(d_input, d_output, MATRIX_SIZE, MATRIX_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float tiled_time = total_time / NUM_ITERATIONS;
    
    // Calculate bandwidth
    float data_gb = 2 * matrix_size / 1e9; // Read + Write
    
    printf("Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Naive transpose:       %.3f ms (%.1f GB/s)\n", 
           naive_time, data_gb / (naive_time / 1000.0));
    printf("Tiled transpose:       %.3f ms (%.1f GB/s) [%.1fx %s]\n", 
           tiled_time, data_gb / (tiled_time / 1000.0),
           fabs(naive_time / tiled_time), tiled_time < naive_time ? "faster" : "slower");
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void testConstantMemory() {
    printf("=== Constant Memory Performance Test ===\n");
    
    float *d_data, *d_lookup;
    size_t data_size = VECTOR_SIZE * sizeof(float);
    size_t lookup_size = 256 * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_lookup, lookup_size));
    
    // Initialize lookup table in constant memory
    std::vector<float> h_lookup(256);
    for (int i = 0; i < 256; i++) {
        h_lookup[i] = sinf(i * 0.1f);
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(const_lookup_table, h_lookup.data(), lookup_size));
    CUDA_CHECK(cudaMemcpy(d_lookup, h_lookup.data(), lookup_size, cudaMemcpyHostToDevice));
    
    // Initialize test data
    std::vector<float> h_data(VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((VECTOR_SIZE + block.x - 1) / block.x);
    
    CudaTimer timer;
    
    // Test constant memory lookup
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        constantMemoryKernel<<<grid, block>>>(d_data, VECTOR_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float const_time = total_time / NUM_ITERATIONS;
    
    // Reset data for fair comparison
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice));
    
    // Test global memory lookup
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        globalMemoryLookupKernel<<<grid, block>>>(d_data, d_lookup, VECTOR_SIZE);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float global_time = total_time / NUM_ITERATIONS;
    
    printf("Array size: %d elements\n", VECTOR_SIZE);
    printf("Constant memory lookup: %.3f ms\n", const_time);
    printf("Global memory lookup:   %.3f ms [%.1fx %s]\n", 
           global_time, fabs(global_time / const_time),
           const_time < global_time ? "slower" : "faster");
    printf("Constant memory speedup: %.1fx\n", global_time / const_time);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_lookup));
}

void testCacheOptimization() {
    printf("=== Cache Optimization Test (Matrix Multiplication) ===\n");
    
    float *d_A, *d_B, *d_C;
    size_t matrix_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_size));
    
    // Initialize matrices
    std::vector<float> h_A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> h_B(MATRIX_SIZE * MATRIX_SIZE);
    
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_size, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((MATRIX_SIZE + block.x - 1) / block.x, (MATRIX_SIZE + block.y - 1) / block.y);
    dim3 tile_block(TILE_SIZE, TILE_SIZE);
    dim3 tile_grid((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    CudaTimer timer;
    
    // Test naive matrix multiplication
    timer.startTimer();
    naiveMatrixMul<<<grid, block>>>(d_A, d_B, d_C, MATRIX_SIZE);
    cudaDeviceSynchronize();
    float naive_time = timer.endTimer();
    
    // Test tiled matrix multiplication
    timer.startTimer();
    tiledMatrixMul<<<tile_grid, tile_block>>>(d_A, d_B, d_C, MATRIX_SIZE);
    cudaDeviceSynchronize();
    float tiled_time = timer.endTimer();
    
    // Calculate GFLOPS
    double ops = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE; // Multiply-add operations
    double naive_gflops = ops / (naive_time / 1000.0) / 1e9;
    double tiled_gflops = ops / (tiled_time / 1000.0) / 1e9;
    
    printf("Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Naive matrix multiply:  %.3f ms (%.1f GFLOPS)\n", naive_time, naive_gflops);
    printf("Tiled matrix multiply:  %.3f ms (%.1f GFLOPS) [%.1fx %s]\n", 
           tiled_time, tiled_gflops, fabs(naive_time / tiled_time),
           tiled_time < naive_time ? "faster" : "slower");
    printf("Cache optimization speedup: %.1fx\n", naive_time / tiled_time);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void testDataLayoutOptimization() {
    printf("=== Data Layout Optimization Test (SoA vs AoS) ===\n");
    
    int n_particles = 1024 * 1024;
    float dt = 0.01f;
    
    // SoA layout
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    size_t particle_size = n_particles * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_x, particle_size));
    CUDA_CHECK(cudaMalloc(&d_y, particle_size));
    CUDA_CHECK(cudaMalloc(&d_z, particle_size));
    CUDA_CHECK(cudaMalloc(&d_vx, particle_size));
    CUDA_CHECK(cudaMalloc(&d_vy, particle_size));
    CUDA_CHECK(cudaMalloc(&d_vz, particle_size));
    
    // AoS layout
    ParticleAoS *d_particles_aos;
    CUDA_CHECK(cudaMalloc(&d_particles_aos, n_particles * sizeof(ParticleAoS)));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((n_particles + block.x - 1) / block.x);
    
    CudaTimer timer;
    
    // Test SoA layout performance
    float total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        updateParticlesSoA<<<grid, block>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, n_particles);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float soa_time = total_time / NUM_ITERATIONS;
    
    // Test AoS layout performance
    total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        timer.startTimer();
        updateParticlesAoS<<<grid, block>>>(d_particles_aos, dt, n_particles);
        cudaDeviceSynchronize();
        total_time += timer.endTimer();
    }
    float aos_time = total_time / NUM_ITERATIONS;
    
    printf("Particle count: %d\n", n_particles);
    printf("SoA layout:  %.3f ms\n", soa_time);
    printf("AoS layout:  %.3f ms [%.1fx %s]\n", 
           aos_time, fabs(aos_time / soa_time),
           soa_time < aos_time ? "slower" : "faster");
    printf("SoA speedup: %.1fx\n", aos_time / soa_time);
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_particles_aos));
}

void printOptimizationSummary() {
    printf("=== Memory Optimization Summary ===\n");
    printf("Key Takeaways:\n\n");
    
    printf("1. Memory Coalescing:\n");
    printf("   - Sequential access patterns achieve highest bandwidth\n");
    printf("   - Strided access severely reduces performance\n");
    printf("   - Vectorized loads (float4) can improve bandwidth utilization\n\n");
    
    printf("2. Shared Memory Optimization:\n");
    printf("   - Tiling techniques improve cache reuse\n");
    printf("   - Avoid bank conflicts with proper padding\n");
    printf("   - Shared memory acts as user-managed cache\n\n");
    
    printf("3. Constant Memory:\n");
    printf("   - Excellent for broadcast reads (same data for all threads)\n");
    printf("   - Limited to 64KB but cached effectively\n");
    printf("   - Significant speedup for lookup tables\n\n");
    
    printf("4. Cache Optimization:\n");
    printf("   - Tiling improves temporal and spatial locality\n");
    printf("   - Shared memory reduces global memory pressure\n");
    printf("   - Block computation to fit cache hierarchy\n\n");
    
    printf("5. Data Layout:\n");
    printf("   - Structure of Arrays (SoA) enables coalescing\n");
    printf("   - Array of Structures (AoS) causes strided access\n");
    printf("   - Choose layout based on access patterns\n\n");
}

int main() {
    printf("=== CUDA Memory Optimization Comprehensive Test ===\n\n");
    
    // Run all optimization tests
    testMemoryCoalescing();
    testSharedMemoryOptimization();
    testConstantMemory();
    testCacheOptimization();
    testDataLayoutOptimization();
    
    // Print summary
    printOptimizationSummary();
    
    printf("=== Memory Optimization Tests Complete ===\n");
    printf("Use profiling tools (ncu, nsys) for detailed analysis!\n");
    
    return 0;
}