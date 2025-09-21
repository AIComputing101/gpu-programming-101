#include <hip/hip_runtime.h>
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
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        // Each thread accesses consecutive memory locations
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Strided memory access (inefficient)
__global__ void stridedAccess(float *data, int n, int stride) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int strided_idx = idx * stride;
    if (strided_idx < n) {
        // Threads access memory with large gaps
        data[strided_idx] = data[strided_idx] * 2.0f + 1.0f;
    }
}

// Misaligned memory access (inefficient)
__global__ void misalignedAccess(float *data, int n, int offset) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x + offset;
    if (idx < n) {
        // Access pattern is shifted, breaking natural alignment
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Structure of Arrays kernel (good coalescing)
__global__ void updateParticlesSoA(ParticlesSoA particles, int n, float dt) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

// Platform-specific optimized versions
#ifdef __HIP_PLATFORM_AMD__
// AMD GCN/RDNA optimized coalesced access
__global__ void coalescedAccessAMD(float *data, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        // Optimized for wavefront memory access patterns
        float value = data[idx];
        value = value * 2.0f + 1.0f;
        data[idx] = value;
    }
}

// AMD-optimized SoA particle update with wavefront considerations
__global__ void updateParticlesSoAAMD(ParticlesSoA particles, int n, float dt) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        // Wavefront-optimized memory access
        float x = particles.x[idx];
        float y = particles.y[idx];
        float z = particles.z[idx];
        float vx = particles.vx[idx];
        float vy = particles.vy[idx];
        float vz = particles.vz[idx];
        
        // Update positions
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
        
        // Update velocities with damping
        vx *= 0.99f;
        vy *= 0.99f;
        vz *= 0.99f;
        
        // Write back
        particles.x[idx] = x;
        particles.y[idx] = y;
        particles.z[idx] = z;
        particles.vx[idx] = vx;
        particles.vy[idx] = vy;
        particles.vz[idx] = vz;
    }
}

#elif defined(__HIP_PLATFORM_NVIDIA__)
// NVIDIA-optimized coalesced access using texture cache
__global__ void coalescedAccessNVIDIA(float *data, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        // Use texture cache for reads
        float value = __ldg(&data[idx]);
        value = value * 2.0f + 1.0f;
        data[idx] = value;
    }
}
#endif

// Vectorized access using float4 (best coalescing)
__global__ void vectorizedAccess(float4 *data, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class MemoryBenchmark {
private:
    float *d_data;
    size_t size;
    hipEvent_t start, stop;
    
public:
    MemoryBenchmark(size_t n) : size(n * sizeof(float)) {
        HIP_CHECK(hipMalloc(&d_data, size));
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Initialize data
        float *h_data = new float[n];
        for (size_t i = 0; i < n; i++) {
            h_data[i] = static_cast<float>(i);
        }
        HIP_CHECK(hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice));
        delete[] h_data;
    }
    
    ~MemoryBenchmark() {
        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    float testCoalesced(int blocks, int threads) {
        HIP_CHECK(hipEventRecord(start));
        coalescedAccess<<<blocks, threads>>>(d_data, size / sizeof(float));
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
        return time;
    }
    
    float testStrided(int blocks, int threads, int stride) {
        HIP_CHECK(hipEventRecord(start));
        stridedAccess<<<blocks, threads>>>(d_data, size / sizeof(float), stride);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
        return time;
    }
    
    float testMisaligned(int blocks, int threads, int offset) {
        HIP_CHECK(hipEventRecord(start));
        misalignedAccess<<<blocks, threads>>>(d_data, size / sizeof(float), offset);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
        return time;
    }
    
    float testPlatformOptimized(int blocks, int threads) {
#ifdef __HIP_PLATFORM_AMD__
        HIP_CHECK(hipEventRecord(start));
        coalescedAccessAMD<<<blocks, threads>>>(d_data, size / sizeof(float));
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
#elif defined(__HIP_PLATFORM_NVIDIA__)
        HIP_CHECK(hipEventRecord(start));
        coalescedAccessNVIDIA<<<blocks, threads>>>(d_data, size / sizeof(float));
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
#else
        return testCoalesced(blocks, threads);
#endif
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
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
    printf("=== HIP Memory Coalescing Benchmarks ===\n");
    
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
    float optimized_time = benchmark.testPlatformOptimized(blocks, threads);
    
    printf("Array size: %zu MB\n", (elements * sizeof(float)) / (1024 * 1024));
    printf("Threads: %d, Blocks: %d\n", threads, blocks);
    printf("\nAccess Pattern Results:\n");
    printf("%-20s %10s %12s\n", "Pattern", "Time (ms)", "Bandwidth (GB/s)");
    printf("%-20s %10.3f %12.2f\n", "Coalesced", coalesced_time, benchmark.getBandwidth(coalesced_time));
    printf("%-20s %10.3f %12.2f\n", "Platform-optimized", optimized_time, benchmark.getBandwidth(optimized_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=2", strided2_time, benchmark.getBandwidth(strided2_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=4", strided4_time, benchmark.getBandwidth(strided4_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=8", strided8_time, benchmark.getBandwidth(strided8_time));
    printf("%-20s %10.3f %12.2f\n", "Stride=32", strided32_time, benchmark.getBandwidth(strided32_time));
    printf("%-20s %10.3f %12.2f\n", "Misaligned", misaligned_time, benchmark.getBandwidth(misaligned_time));
    
    printf("\nPerformance Impact:\n");
    printf("Optimized vs Coalesced: %.2fx %s\n", 
           optimized_time / coalesced_time, 
           (optimized_time < coalesced_time) ? "faster" : "slower");
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
    HIP_CHECK(hipMalloc(&soa.x, array_size));
    HIP_CHECK(hipMalloc(&soa.y, array_size));
    HIP_CHECK(hipMalloc(&soa.z, array_size));
    HIP_CHECK(hipMalloc(&soa.vx, array_size));
    HIP_CHECK(hipMalloc(&soa.vy, array_size));
    HIP_CHECK(hipMalloc(&soa.vz, array_size));
    HIP_CHECK(hipMalloc(&soa.mass, array_size));
    
    // Setup AoS
    ParticleAoS *aos;
    size_t aos_size = n_particles * sizeof(ParticleAoS);
    HIP_CHECK(hipMalloc(&aos, aos_size));
    
    // Initialize data (simplified - just allocate)
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    const int iterations = 100;
    
    // Benchmark SoA
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        updateParticlesSoA<<<blocks, threads>>>(soa, n_particles, dt);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float soa_time;
    HIP_CHECK(hipEventElapsedTime(&soa_time, start, stop));
    
    // Benchmark AoS
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        updateParticlesAoS<<<blocks, threads>>>(aos, n_particles, dt);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float aos_time;
    HIP_CHECK(hipEventElapsedTime(&aos_time, start, stop));
    
    // Benchmark platform-specific SoA optimization
    float soa_optimized_time = soa_time;
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        updateParticlesSoAAMD<<<blocks, threads>>>(soa, n_particles, dt);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&soa_optimized_time, start, stop));
#endif
    
    printf("Particles: %d\n", n_particles);
    printf("Iterations: %d\n", iterations);
    printf("SoA time: %.3f ms (%.3f ms per iteration)\n", soa_time, soa_time / iterations);
    printf("AoS time: %.3f ms (%.3f ms per iteration)\n", aos_time, aos_time / iterations);
    printf("SoA speedup over AoS: %.2fx\n", aos_time / soa_time);
    
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD-optimized SoA time: %.3f ms (%.3f ms per iteration)\n", 
           soa_optimized_time, soa_optimized_time / iterations);
    printf("AMD optimization speedup: %.2fx\n", soa_time / soa_optimized_time);
#endif
    
    // Memory throughput analysis
    double soa_bytes = 7.0 * array_size * iterations;  // 7 arrays, iterations
    double aos_bytes = 2.0 * aos_size * iterations;    // Read and write structure
    
    printf("\nMemory Analysis:\n");
    printf("SoA bandwidth: %.2f GB/s\n", 
           (soa_bytes / (1024.0 * 1024.0 * 1024.0)) / (soa_time / 1000.0));
    printf("AoS bandwidth: %.2f GB/s\n", 
           (aos_bytes / (1024.0 * 1024.0 * 1024.0)) / (aos_time / 1000.0));
    
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD SoA bandwidth: %.2f GB/s\n", 
           (soa_bytes / (1024.0 * 1024.0 * 1024.0)) / (soa_optimized_time / 1000.0));
#endif
    
    // Cleanup
    HIP_CHECK(hipFree(soa.x)); HIP_CHECK(hipFree(soa.y)); HIP_CHECK(hipFree(soa.z));
    HIP_CHECK(hipFree(soa.vx)); HIP_CHECK(hipFree(soa.vy)); HIP_CHECK(hipFree(soa.vz));
    HIP_CHECK(hipFree(soa.mass)); HIP_CHECK(hipFree(aos));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
}

void runVectorizationBenchmark() {
    printf("\n=== Vectorized Memory Access ===\n");
    
    const size_t elements = 16 * 1024 * 1024;  // 16M floats = 4M float4s
    const size_t float_size = elements * sizeof(float);
    const size_t float4_elements = elements / 4;
    
    float *d_float_data;
    float4 *d_float4_data;
    
    HIP_CHECK(hipMalloc(&d_float_data, float_size));
    HIP_CHECK(hipMalloc(&d_float4_data, float_size));
    
    int threads = 256;
    int float_blocks = (elements + threads - 1) / threads;
    int float4_blocks = (float4_elements + threads - 1) / threads;
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Test scalar access
    HIP_CHECK(hipEventRecord(start));
    coalescedAccess<<<float_blocks, threads>>>(d_float_data, elements);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float scalar_time;
    HIP_CHECK(hipEventElapsedTime(&scalar_time, start, stop));
    
    // Test vectorized access
    HIP_CHECK(hipEventRecord(start));
    vectorizedAccess<<<float4_blocks, threads>>>(d_float4_data, float4_elements);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float vector_time;
    HIP_CHECK(hipEventElapsedTime(&vector_time, start, stop));
    
    double bytes_transferred = 2.0 * float_size;  // Read and write
    
    printf("Array size: %zu MB\n", float_size / (1024 * 1024));
    printf("Scalar access time: %.3f ms\n", scalar_time);
    printf("Vector access time: %.3f ms\n", vector_time);
    printf("Vectorization speedup: %.2fx\n", scalar_time / vector_time);
    
    printf("Scalar bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (scalar_time / 1000.0));
    printf("Vector bandwidth: %.2f GB/s\n", 
           (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (vector_time / 1000.0));
    
    HIP_CHECK(hipFree(d_float_data));
    HIP_CHECK(hipFree(d_float4_data));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

int main() {
    printf("HIP Memory Coalescing Analysis\n");
    printf("==============================\n");
    
    // Get device properties
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    printf("Running on: %s\n", props.name);
    printf("Platform: ");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD ROCm\n");
    printf("Wavefront size: %d\n", props.warpSize);
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA\n");
    printf("Warp size: %d\n", props.warpSize);
#else
    printf("Unknown\n");
#endif
    
    double theoreticalBW = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical memory bandwidth: %.1f GB/s\n", theoreticalBW);
    printf("Memory bus width: %d bits\n", props.memoryBusWidth);
    printf("Memory clock rate: %d MHz\n", props.memoryClockRate / 1000);
    
    // Run benchmarks
    runCoalescingBenchmarks();
    runParticleBenchmarks();
    runVectorizationBenchmark();
    
    // Platform-specific analysis
    printf("\n=== Platform-Specific Memory Optimization ===\n");
#ifdef __HIP_PLATFORM_AMD__
    printf("AMD GCN/RDNA Memory Guidelines:\n");
    printf("✓ Optimize for wavefront-level memory coalescing (64 threads)\n");
    printf("✓ Use Local Data Share (LDS) efficiently\n");
    printf("✓ Consider memory bank conflicts in LDS\n");
    printf("✓ Utilize HBM memory bandwidth effectively\n");
    printf("✓ Be aware of memory channel interleaving\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    printf("NVIDIA CUDA Memory Guidelines:\n");
    printf("✓ Optimize for warp-level memory coalescing (32 threads)\n");
    printf("✓ Use texture cache with __ldg() for read-only data\n");
    printf("✓ Avoid shared memory bank conflicts\n");
    printf("✓ Utilize L1/L2 cache hierarchy effectively\n");
    printf("✓ Consider using unified memory for complex patterns\n");
#endif
    
    // Educational summary
    printf("\n=== Memory Coalescing Guidelines ===\n");
    printf("✓ GOOD PRACTICES:\n");
    printf("  - Access consecutive memory locations within a warp/wavefront\n");
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
    printf("  - Profile with rocprof/nvprof to identify coalescing issues\n");
    printf("  - Restructure data layouts for better access patterns\n");
    printf("  - Leverage platform-specific optimizations\n");
    
    printf("\nHIP memory coalescing analysis completed successfully!\n");
    return 0;
}