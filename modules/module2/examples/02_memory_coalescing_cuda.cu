#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Structure of Arrays (SoA) - Good for coalescing
struct ParticlesSoA {
    float *x, *y, *z;        // Position arrays
    float *vx, *vy, *vz;     // Velocity arrays
    float *mass;             // Mass array
};

// Array of Structures (AoS) - Poor for coalescing
struct ParticleAoS {
    float x, y, z;           // Position
    float vx, vy, vz;        // Velocity
    float mass;              // Mass
};

// Coalesced memory access (efficient)
__global__ void coalescedAccess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread accesses consecutive memory locations
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Strided memory access (inefficient)
__global__ void stridedAccess(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    if (strided_idx < n) {
        // Threads access memory with large gaps
        data[strided_idx] = data[strided_idx] * 2.0f + 1.0f;
    }
}

// Misaligned memory access (inefficient)
__global__ void misalignedAccess(float *data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        // Access pattern is shifted, breaking natural alignment
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Structure of Arrays kernel (good coalescing)
__global__ void updateParticlesSoA(ParticlesSoA particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // All threads access consecutive memory in each array
        particles.x[idx] += particles.vx[idx] * dt;
        particles.y[idx] += particles.vy[idx] * dt;
        particles.z[idx] += particles.vz[idx] * dt;
        
        // Apply simple physics
        particles.vx[idx] *= 0.99f;  // Damping
        particles.vy[idx] *= 0.99f;
        particles.vz[idx] *= 0.99f;
    }
}

// Array of Structures kernel (poor coalescing)
__global__ void updateParticlesAoS(ParticleAoS *particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Threads access scattered memory locations
        particles[idx].x += particles[idx].vx * dt;
        particles[idx].y += particles[idx].vy * dt;
        particles[idx].z += particles[idx].vz * dt;
        
        particles[idx].vx *= 0.99f;
        particles[idx].vy *= 0.99f;
        particles[idx].vz *= 0.99f;
    }
}

// Vectorized access using float4 (best coalescing)
__global__ void vectorizedAccess(float4 *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load 16 bytes (4 floats) in one transaction
        float4 vec = data[idx];
        
        // Process all components
        vec.x = vec.x * 2.0f + 1.0f;
        vec.y = vec.y * 2.0f + 1.0f;
        vec.z = vec.z * 2.0f + 1.0f;
        vec.w = vec.w * 2.0f + 1.0f;
        
        // Store back
        data[idx] = vec;
    }
}

// Memory access pattern visualization
__global__ void visualizeAccessPattern(float *data, int *access_pattern, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Record which thread accesses which memory location
        access_pattern[idx] = idx;
        data[idx] = (float)idx;
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

class MemoryBenchmark {
private:
    float *d_data;
    size_t size;
    cudaEvent_t start, stop;
    
public:
    MemoryBenchmark(size_t n) : size(n * sizeof(float)) {
        CUDA_CHECK(cudaMalloc(&d_data, size));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Initialize data
        float *h_data = new float[n];
        for (size_t i = 0; i < n; i++) {
            h_data[i] = static_cast<float>(i);
        }
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        delete[] h_data;
    }
    
    ~MemoryBenchmark() {
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    float testCoalesced(int blocks, int threads) {
        CUDA_CHECK(cudaEventRecord(start));
        coalescedAccess<<<blocks, threads>>>(d_data, size / sizeof(float));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        return time;
    }
    
    float testStrided(int blocks, int threads, int stride) {
        CUDA_CHECK(cudaEventRecord(start));
        stridedAccess<<<blocks, threads>>>(d_data, size / sizeof(float), stride);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        return time;
    }
    
    float testMisaligned(int blocks, int threads, int offset) {
        CUDA_CHECK(cudaEventRecord(start));
        misalignedAccess<<<blocks, threads>>>(d_data, size / sizeof(float), offset);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        return time;
    }
    
    double getBandwidth(float time_ms) {
        // Bandwidth = bytes transferred / time
        // We read and write each element once
        double bytes = 2.0 * size;
        return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    }
};

void runCoalescingBenchmarks() {
    printf("=== Memory Coalescing Benchmarks ===\n");
    
    const size_t elements = 64 * 1024 * 1024;  // 64M elements
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    
    MemoryBenchmark benchmark(elements);
    
    // Test different access patterns
    float coalesced_time = benchmark.testCoalesced(blocks, threads);
    float strided2_time = benchmark.testStrided(blocks, threads, 2);
    float strided4_time = benchmark.testStrided(blocks, threads, 4);
    float strided8_time = benchmark.testStrided(blocks, threads, 8);
    float strided32_time = benchmark.testStrided(blocks, threads, 32);
    float misaligned_time = benchmark.testMisaligned(blocks, threads, 1);
    
    printf("Array size: %zu MB\n", (elements * sizeof(float)) / (1024 * 1024));
    printf("Threads: %d, Blocks: %d\n", threads, blocks);
    printf("\nAccess Pattern Results:\n");
    printf("%-20s %10s %12s\n", "Pattern", "Time (ms)", "Bandwidth (GB/s)");
    printf("%-20s %10.3f %12.2f\n", "Coalesced", coalesced_time, benchmark.getBandwidth(coalesced_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=2", strided2_time, benchmark.getBandwidth(strided2_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=4", strided4_time, benchmark.getBandwidth(strided4_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=8", strided8_time, benchmark.getBandwidth(strided8_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=32", strided32_time, benchmark.getBandwidth(strided32_time));
    printf("%-20s %10.3f %12.2f\n", "Misaligned", misaligned_time, benchmark.getBandwidth(misaligned_time));
    
    printf("\nPerformance Impact:\n");
    printf("Stride=2 vs Coalesced: %.2fx slower\n", strided2_time / coalesced_time);
    printf("Stride=32 vs Coalesced: %.2fx slower\n", strided32_time / coalesced_time);
    printf("Misaligned vs Coalesced: %.2fx slower\n", misaligned_time / coalesced_time);
}

void runParticleBenchmarks() {
    printf("\n=== Particle System: AoS vs SoA ===\n");
    
    const int n_particles = 1024 * 1024;  // 1M particles
    const float dt = 0.016f;  // 60 FPS
    
    // Setup SoA
    ParticlesSoA soa;
    size_t array_size = n_particles * sizeof(float);
    CUDA_CHECK(cudaMalloc(&soa.x, array_size));
    CUDA_CHECK(cudaMalloc(&soa.y, array_size));
    CUDA_CHECK(cudaMalloc(&soa.z, array_size));
    CUDA_CHECK(cudaMalloc(&soa.vx, array_size));
    CUDA_CHECK(cudaMalloc(&soa.vy, array_size));
    CUDA_CHECK(cudaMalloc(&soa.vz, array_size));
    CUDA_CHECK(cudaMalloc(&soa.mass, array_size));
    
    // Setup AoS
    ParticleAoS *aos;
    size_t aos_size = n_particles * sizeof(ParticleAoS);
    CUDA_CHECK(cudaMalloc(&aos, aos_size));
    
    // Initialize data (simplified - just allocate)
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Benchmark SoA
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        updateParticlesSoA<<<blocks, threads>>>(soa, n_particles, dt);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float soa_time;
    CUDA_CHECK(cudaEventElapsedTime(&soa_time, start, stop));
    
    // Benchmark AoS
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        updateParticlesAoS<<<blocks, threads>>>(aos, n_particles, dt);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float aos_time;
    CUDA_CHECK(cudaEventElapsedTime(&aos_time, start, stop));
    
    printf("Particles: %d\n", n_particles);
    printf("Iterations: 100\n");
    printf("SoA time: %.3f ms (%.3f ms per iteration)\n", soa_time, soa_time / 100.0f);
    printf("AoS time: %.3f ms (%.3f ms per iteration)\n", aos_time, aos_time / 100.0f);
    printf("SoA speedup: %.2fx\n", aos_time / soa_time);
    
    // Memory throughput analysis
    double soa_bytes = 7.0 * array_size * 100;  // 7 arrays, 100 iterations
    double aos_bytes = 2.0 * aos_size * 100;    // Read and write structure, 100 iterations
    
    printf("\nMemory Analysis:\n");
    printf("SoA bandwidth: %.2f GB/s\n", (soa_bytes / (1024.0 * 1024.0 * 1024.0)) / (soa_time / 1000.0));
    printf("AoS bandwidth: %.2f GB/s\n", (aos_bytes / (1024.0 * 1024.0 * 1024.0)) / (aos_time / 1000.0));
    
    // Cleanup
    cudaFree(soa.x); cudaFree(soa.y); cudaFree(soa.z);
    cudaFree(soa.vx); cudaFree(soa.vy); cudaFree(soa.vz);
    cudaFree(soa.mass); cudaFree(aos);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void runVectorizationBenchmark() {
    printf("\n=== Vectorized Memory Access ===\n");
    
    const size_t elements = 16 * 1024 * 1024;  // 16M floats = 4M float4s
    const size_t float_size = elements * sizeof(float);
    const size_t float4_elements = elements / 4;
    
    float *d_float_data;
    float4 *d_float4_data;
    
    CUDA_CHECK(cudaMalloc(&d_float_data, float_size));
    CUDA_CHECK(cudaMalloc(&d_float4_data, float_size));
    
    int threads = 256;
    int float_blocks = (elements + threads - 1) / threads;
    int float4_blocks = (float4_elements + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test scalar access
    CUDA_CHECK(cudaEventRecord(start));
    coalescedAccess<<<float_blocks, threads>>>(d_float_data, elements);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float scalar_time;
    CUDA_CHECK(cudaEventElapsedTime(&scalar_time, start, stop));
    
    // Test vectorized access
    CUDA_CHECK(cudaEventRecord(start));
    vectorizedAccess<<<float4_blocks, threads>>>(d_float4_data, float4_elements);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float vector_time;
    CUDA_CHECK(cudaEventElapsedTime(&vector_time, start, stop));
    
    double bytes_transferred = 2.0 * float_size;  // Read and write
    
    printf("Array size: %zu MB\n", float_size / (1024 * 1024));
    printf("Scalar access time: %.3f ms\n", scalar_time);
    printf("Vector access time: %.3f ms\n", vector_time);
    printf("Vectorization speedup: %.2fx\n", scalar_time / vector_time);
    
    printf("Scalar bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (scalar_time / 1000.0));
    printf("Vector bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (vector_time / 1000.0));
    
    cudaFree(d_float_data);
    cudaFree(d_float4_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Memory Coalescing Analysis\n");
    printf("===============================\n");
    
    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    printf("Global memory bandwidth: %.1f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
    printf("Memory bus width: %d bits\n", props.memoryBusWidth);
    printf("Memory clock rate: %d MHz\n", props.memoryClockRate / 1000);
    printf("Warp size: %d threads\n", props.warpSize);
    
    // Run benchmarks
    runCoalescingBenchmarks();
    runParticleBenchmarks();
    runVectorizationBenchmark();
    
    // Educational summary
    printf("\n=== Memory Coalescing Guidelines ===\n");
    printf("✓ GOOD PRACTICES:\n");
    printf("  - Access consecutive memory locations within a warp\n");
    printf("  - Use Structure of Arrays (SoA) for parallel processing\n");
    printf("  - Align data structures to memory boundaries\n");
    printf("  - Use vectorized types (float2, float4) when appropriate\n");
    printf("  - Ensure memory transactions are naturally aligned\n");
    
    printf("\n✗ AVOID:\n");
    printf("  - Large stride access patterns\n");
    printf("  - Array of Structures (AoS) for compute-heavy kernels\n");
    printf("  - Misaligned memory accesses\n");
    printf("  - Random or scattered memory access patterns\n");
    
    printf("\n💡 OPTIMIZATION TIPS:\n");
    printf("  - Use 128-byte memory transactions when possible\n");
    printf("  - Consider memory layout during algorithm design\n");
    printf("  - Profile with nvprof to identify coalescing issues\n");
    printf("  - Restructure data layouts for better access patterns\n");
    
    printf("\nMemory coalescing analysis completed successfully!\n");
    return 0;
}