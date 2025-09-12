#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <complex>
#include <memory>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult result = call; \
    if (result != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error: " << result << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

// N-body simulation kernel
__global__ void nbody_kernel(float4* positions, float4* velocities, float4* forces, int n, float dt, float eps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    float4 pos = positions[tid];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Calculate forces from all other particles
    for (int j = 0; j < n; j++) {
        if (j != tid) {
            float4 other_pos = positions[j];
            
            float dx = other_pos.x - pos.x;
            float dy = other_pos.y - pos.y;
            float dz = other_pos.z - pos.z;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + eps*eps;
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            
            float f = other_pos.w * inv_dist3; // mass * 1/r^3
            
            force.x += f * dx;
            force.y += f * dy;
            force.z += f * dz;
        }
    }
    
    forces[tid] = force;
    
    // Update velocity and position
    float4 vel = velocities[tid];
    vel.x += force.x * dt / pos.w; // F/m * dt
    vel.y += force.y * dt / pos.w;
    vel.z += force.z * dt / pos.w;
    
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    velocities[tid] = vel;
    positions[tid] = pos;
}

// Optimized N-body with shared memory
__global__ void nbody_shared_kernel(float4* positions, float4* velocities, float4* forces, int n, float dt, float eps) {
    extern __shared__ float4 shared_pos[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    float4 pos = positions[tid];
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        
        // Load tile into shared memory
        if (idx < n) {
            shared_pos[threadIdx.x] = positions[idx];
        } else {
            shared_pos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        
        // Compute forces for this tile
        for (int j = 0; j < blockDim.x; j++) {
            int other_idx = tile * blockDim.x + j;
            if (other_idx < n && other_idx != tid) {
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
        }
        __syncthreads();
    }
    
    forces[tid] = force;
    
    // Update velocity and position
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

// Monte Carlo Pi estimation
__global__ void monte_carlo_pi_kernel(curandState* states, int* hits, int n_samples_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState local_state = states[tid];
    int local_hits = 0;
    
    for (int i = 0; i < n_samples_per_thread; i++) {
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);
        
        if (x*x + y*y <= 1.0f) {
            local_hits++;
        }
    }
    
    hits[tid] = local_hits;
    states[tid] = local_state;
}

__global__ void setup_curand_states(curandState* states, unsigned long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Heat equation solver using finite differences
__global__ void heat_equation_kernel(float* u_new, const float* u_old, int nx, int ny, float alpha, float dt, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip boundary
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < nx-1 && j < ny-1) {
        int idx = j * nx + i;
        
        float u_center = u_old[idx];
        float u_left = u_old[idx - 1];
        float u_right = u_old[idx + 1];
        float u_down = u_old[idx - nx];
        float u_up = u_old[idx + nx];
        
        float d2u_dx2 = (u_left - 2.0f * u_center + u_right) / (dx * dx);
        float d2u_dy2 = (u_down - 2.0f * u_center + u_up) / (dy * dy);
        
        u_new[idx] = u_center + alpha * dt * (d2u_dx2 + d2u_dy2);
    }
}

// Wave equation solver
__global__ void wave_equation_kernel(float* u_new, const float* u, const float* u_old, 
                                     int nx, int ny, float c, float dt, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < nx-1 && j < ny-1) {
        int idx = j * nx + i;
        
        float u_center = u[idx];
        float u_left = u[idx - 1];
        float u_right = u[idx + 1];
        float u_down = u[idx - nx];
        float u_up = u[idx + nx];
        
        float d2u_dx2 = (u_left - 2.0f * u_center + u_right) / (dx * dx);
        float d2u_dy2 = (u_down - 2.0f * u_center + u_up) / (dy * dy);
        
        float c_sq_dt_sq = c * c * dt * dt;
        u_new[idx] = 2.0f * u_center - u_old[idx] + c_sq_dt_sq * (d2u_dx2 + d2u_dy2);
    }
}

// Molecular dynamics simulation
struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
};

__global__ void md_lennard_jones_kernel(Particle* particles, int n, float epsilon, float sigma, float cutoff_sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = particles[i].position;
    
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float3 pos_j = particles[j].position;
            
            float dx = pos_j.x - pos_i.x;
            float dy = pos_j.y - pos_i.y;
            float dz = pos_j.z - pos_i.z;
            
            float r_sq = dx*dx + dy*dy + dz*dz;
            
            if (r_sq < cutoff_sq && r_sq > 1e-6f) {
                float sigma_sq = sigma * sigma;
                float sigma_6 = sigma_sq * sigma_sq * sigma_sq;
                float sigma_12 = sigma_6 * sigma_6;
                
                float r_6 = r_sq * r_sq * r_sq;
                float r_12 = r_6 * r_6;
                
                float f_magnitude = 24.0f * epsilon * (2.0f * sigma_12 / r_12 - sigma_6 / r_6) / r_sq;
                
                force.x += f_magnitude * dx;
                force.y += f_magnitude * dy;
                force.z += f_magnitude * dz;
            }
        }
    }
    
    particles[i].force = force;
}

__global__ void md_update_positions_kernel(Particle* particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    Particle& p = particles[i];
    
    // Velocity Verlet integration
    p.velocity.x += 0.5f * p.force.x * dt / p.mass;
    p.velocity.y += 0.5f * p.force.y * dt / p.mass;
    p.velocity.z += 0.5f * p.force.z * dt / p.mass;
    
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;
}

class ScientificComputingDemo {
private:
    cublasHandle_t cublas_handle;
    
public:
    ScientificComputingDemo() {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
    }
    
    ~ScientificComputingDemo() {
        cublasDestroy(cublas_handle);
    }
    
    void demonstrateNBodySimulation() {
        std::cout << "\n=== N-Body Simulation ===" << std::endl;
        
        const int n = 2048;
        const float dt = 0.01f;
        const float eps = 0.1f;
        const int n_steps = 100;
        
        // Allocate memory
        float4 *d_positions, *d_velocities, *d_forces;
        CHECK_CUDA(cudaMalloc(&d_positions, n * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_velocities, n * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_forces, n * sizeof(float4)));
        
        // Initialize particles
        std::vector<float4> positions(n), velocities(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> mass_dist(0.1f, 2.0f);
        
        for (int i = 0; i < n; i++) {
            positions[i] = make_float4(pos_dist(gen), pos_dist(gen), pos_dist(gen), mass_dist(gen));
            velocities[i] = make_float4(vel_dist(gen), vel_dist(gen), vel_dist(gen), 0.0f);
        }
        
        CHECK_CUDA(cudaMemcpy(d_positions, positions.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_velocities, velocities.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        Timer timer;
        
        // Test naive implementation
        timer.start();
        for (int step = 0; step < n_steps; step++) {
            nbody_kernel<<<grid_size, block_size>>>(d_positions, d_velocities, d_forces, n, dt, eps);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        double naive_time = timer.elapsed();
        
        // Reset positions and velocities
        CHECK_CUDA(cudaMemcpy(d_positions, positions.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_velocities, velocities.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        
        // Test optimized implementation with shared memory
        size_t shared_mem_size = block_size.x * sizeof(float4);
        timer.start();
        for (int step = 0; step < n_steps; step++) {
            nbody_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(d_positions, d_velocities, d_forces, n, dt, eps);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        double optimized_time = timer.elapsed();
        
        std::cout << "Naive N-body time: " << naive_time << " ms" << std::endl;
        std::cout << "Optimized N-body time: " << optimized_time << " ms" << std::endl;
        std::cout << "Speedup: " << naive_time / optimized_time << "x" << std::endl;
        
        // Calculate performance metrics
        double flops_per_step = (double)n * n * 20; // Approximate FLOPs per interaction
        double total_flops = flops_per_step * n_steps;
        double gflops_optimized = total_flops / (optimized_time * 1e6);
        std::cout << "Performance: " << gflops_optimized << " GFLOPS" << std::endl;
        
        cudaFree(d_positions);
        cudaFree(d_velocities);
        cudaFree(d_forces);
    }
    
    void demonstrateMonteCarloPi() {
        std::cout << "\n=== Monte Carlo Pi Estimation ===" << std::endl;
        
        const int n_threads = 1024 * 512;
        const int n_samples_per_thread = 1000;
        const long long total_samples = (long long)n_threads * n_samples_per_thread;
        
        // Allocate memory
        curandState *d_states;
        int *d_hits;
        CHECK_CUDA(cudaMalloc(&d_states, n_threads * sizeof(curandState)));
        CHECK_CUDA(cudaMalloc(&d_hits, n_threads * sizeof(int)));
        
        dim3 block_size(256);
        dim3 grid_size((n_threads + block_size.x - 1) / block_size.x);
        
        Timer timer;
        timer.start();
        
        // Initialize random states
        setup_curand_states<<<grid_size, block_size>>>(d_states, time(nullptr), n_threads);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Run Monte Carlo simulation
        monte_carlo_pi_kernel<<<grid_size, block_size>>>(d_states, d_hits, n_samples_per_thread);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Reduce results using Thrust
        thrust::device_ptr<int> thrust_hits(d_hits);
        int total_hits = thrust::reduce(thrust_hits, thrust_hits + n_threads);
        
        double elapsed = timer.elapsed();
        
        double pi_estimate = 4.0 * total_hits / total_samples;
        double error = std::abs(pi_estimate - M_PI);
        
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "Estimated Pi: " << pi_estimate << std::endl;
        std::cout << "Actual Pi: " << M_PI << std::endl;
        std::cout << "Error: " << error << std::endl;
        std::cout << "Time: " << elapsed << " ms" << std::endl;
        std::cout << "Samples per second: " << total_samples / (elapsed / 1000.0) / 1e9 << " billion" << std::endl;
        
        cudaFree(d_states);
        cudaFree(d_hits);
    }
    
    void demonstratePDESolver() {
        std::cout << "\n=== PDE Solver (Heat Equation) ===" << std::endl;
        
        const int nx = 512, ny = 512;
        const float lx = 1.0f, ly = 1.0f;
        const float dx = lx / (nx - 1);
        const float dy = ly / (ny - 1);
        const float alpha = 0.01f; // Thermal diffusivity
        const float dt = 0.001f;
        const int n_steps = 1000;
        
        // Allocate memory
        float *d_u, *d_u_new;
        CHECK_CUDA(cudaMalloc(&d_u, nx * ny * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_u_new, nx * ny * sizeof(float)));
        
        // Initialize temperature field
        std::vector<float> u_init(nx * ny, 0.0f);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float x = i * dx;
                float y = j * dy;
                
                // Initial condition: Gaussian hot spot in center
                float x_center = 0.5f, y_center = 0.5f;
                float sigma = 0.1f;
                u_init[j * nx + i] = 100.0f * exp(-((x - x_center) * (x - x_center) + 
                                                   (y - y_center) * (y - y_center)) / (2 * sigma * sigma));
            }
        }
        
        CHECK_CUDA(cudaMemcpy(d_u, u_init.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 block_size(16, 16);
        dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
        
        Timer timer;
        timer.start();
        
        // Time stepping
        for (int step = 0; step < n_steps; step++) {
            heat_equation_kernel<<<grid_size, block_size>>>(d_u_new, d_u, nx, ny, alpha, dt, dx, dy);
            std::swap(d_u, d_u_new);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        double elapsed = timer.elapsed();
        
        std::cout << "Grid size: " << nx << "x" << ny << std::endl;
        std::cout << "Time steps: " << n_steps << std::endl;
        std::cout << "Simulation time: " << elapsed << " ms" << std::endl;
        
        // Calculate performance metrics
        double grid_points_per_step = (double)(nx - 2) * (ny - 2);
        double total_grid_updates = grid_points_per_step * n_steps;
        double updates_per_second = total_grid_updates / (elapsed / 1000.0);
        std::cout << "Grid point updates per second: " << updates_per_second / 1e9 << " billion" << std::endl;
        
        cudaFree(d_u);
        cudaFree(d_u_new);
    }
    
    void demonstrateFFT() {
        std::cout << "\n=== Fast Fourier Transform ===" << std::endl;
        
        const int n = 1024 * 1024;
        
        // Allocate memory
        cufftComplex *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(cufftComplex)));
        
        // Initialize with a test signal
        std::vector<cufftComplex> h_data(n);
        for (int i = 0; i < n; i++) {
            float t = 2.0f * M_PI * i / n;
            h_data[i].x = cosf(10 * t) + 0.5f * cosf(25 * t); // Real part
            h_data[i].y = 0.0f; // Imaginary part
        }
        
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), n * sizeof(cufftComplex), cudaMemcpyHostToDevice));
        
        // Create FFT plan
        cufftHandle plan;
        CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
        
        Timer timer;
        timer.start();
        
        // Forward FFT
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        double forward_time = timer.elapsed();
        
        timer.start();
        
        // Inverse FFT
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        double inverse_time = timer.elapsed();
        
        std::cout << "FFT size: " << n << std::endl;
        std::cout << "Forward FFT time: " << forward_time << " ms" << std::endl;
        std::cout << "Inverse FFT time: " << inverse_time << " ms" << std::endl;
        
        // Calculate performance
        double flops_estimate = 5.0 * n * log2(n); // Rough estimate for complex FFT
        double gflops_forward = flops_estimate / (forward_time * 1e6);
        std::cout << "Forward FFT performance: " << gflops_forward << " GFLOPS" << std::endl;
        
        cufftDestroy(plan);
        cudaFree(d_data);
    }
};

int main() {
    std::cout << "CUDA Scientific Computing Demo" << std::endl;
    std::cout << "==============================" << std::endl;
    
    ScientificComputingDemo demo;
    
    demo.demonstrateNBodySimulation();
    demo.demonstrateMonteCarloPi();
    demo.demonstratePDESolver();
    demo.demonstrateFFT();
    
    return 0;
}