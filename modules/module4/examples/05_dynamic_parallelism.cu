#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define MAX_DEPTH 6
#define MIN_SIZE 1024
#define BLOCK_SIZE 256

// Forward declarations for device functions
__device__ void deviceQuicksort(float *data, int left, int right, int depth);
__device__ int devicePartition(float *data, int left, int right);

// Simple parallel reduction with dynamic parallelism
__global__ void dynamicReduction(float *input, float *output, int n, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Base case: use traditional reduction
    if (n <= blockDim.x || depth <= 0) {
        extern __shared__ float sdata[];
        
        sdata[threadIdx.x] = (tid < n) ? input[tid] : 0.0f;
        __syncthreads();
        
        // Traditional reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && tid + stride < n) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = sdata[0];
        }
        return;
    }
    
    // Recursive case: launch child kernels
    if (tid == 0) {
        int halfSize = n / 2;
        
        // Launch two child kernels for each half
        dim3 childGrid((halfSize + blockDim.x - 1) / blockDim.x);
        dim3 childBlock(blockDim.x);
        
        float *temp1, *temp2;
        cudaMalloc(&temp1, sizeof(float));
        cudaMalloc(&temp2, sizeof(float));
        
        dynamicReduction<<<childGrid, childBlock, blockDim.x * sizeof(float)>>>(
            input, temp1, halfSize, depth - 1);
        
        dynamicReduction<<<childGrid, childBlock, blockDim.x * sizeof(float)>>>(
            input + halfSize, temp2, n - halfSize, depth - 1);
        
        cudaDeviceSynchronize(); // Wait for child kernels
        
        // Combine results
        float result1, result2;
        cudaMemcpy(&result1, temp1, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&result2, temp2, sizeof(float), cudaMemcpyDeviceToDevice);
        
        *output = result1 + result2;
        
        cudaFree(temp1);
        cudaFree(temp2);
    }
}

// Adaptive mesh refinement kernel
__global__ void adaptiveMeshRefinement(float *data, bool *refineFlags, int width, int height, 
                                      int level, int maxLevel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = data[idx];
    
    // Check if this cell needs refinement (simple gradient-based criterion)
    bool needsRefinement = false;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gradient = fabsf(data[idx] - data[idx - 1]) + 
                        fabsf(data[idx] - data[idx + 1]) +
                        fabsf(data[idx] - data[idx - width]) +
                        fabsf(data[idx] - data[idx + width]);
        
        needsRefinement = gradient > 0.5f; // Refinement threshold
    }
    
    refineFlags[idx] = needsRefinement;
    
    // If refinement is needed and we haven't reached max level, launch child kernel
    if (needsRefinement && level < maxLevel && threadIdx.x == 0 && threadIdx.y == 0) {
        // Create refined grid (2x2 subdivision)
        float *refinedData;
        int refinedSize = 4 * sizeof(float);
        cudaMalloc(&refinedData, refinedSize);
        
        // Initialize refined cells
        refinedData[0] = value + 0.1f * (rand() % 100 - 50) / 100.0f;
        refinedData[1] = value + 0.1f * (rand() % 100 - 50) / 100.0f;
        refinedData[2] = value + 0.1f * (rand() % 100 - 50) / 100.0f;
        refinedData[3] = value + 0.1f * (rand() % 100 - 50) / 100.0f;
        
        // Launch child kernel for refined region
        dim3 childBlock(2, 2);
        dim3 childGrid(1);
        
        bool *childFlags;
        cudaMalloc(&childFlags, 4 * sizeof(bool));
        
        adaptiveMeshRefinement<<<childGrid, childBlock>>>(
            refinedData, childFlags, 2, 2, level + 1, maxLevel);
        
        cudaDeviceSynchronize();
        
        // Update original data with refined values (simplified)
        data[idx] = (refinedData[0] + refinedData[1] + refinedData[2] + refinedData[3]) / 4.0f;
        
        cudaFree(refinedData);
        cudaFree(childFlags);
    }
}

// Recursive ray tracing kernel
struct Ray {
    float3 origin;
    float3 direction;
    int depth;
};

struct Sphere {
    float3 center;
    float radius;
    float3 color;
};

__device__ float3 make_float3_device(float x, float y, float z) {
    float3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

__device__ float dot_product(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 subtract_float3(float3 a, float3 b) {
    return make_float3_device(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 add_float3(float3 a, float3 b) {
    return make_float3_device(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 scale_float3(float3 v, float s) {
    return make_float3_device(v.x * s, v.y * s, v.z * s);
}

__device__ bool intersectSphere(const Ray& ray, const Sphere& sphere, float& t) {
    float3 oc = subtract_float3(ray.origin, sphere.center);
    float a = dot_product(ray.direction, ray.direction);
    float b = 2.0f * dot_product(oc, ray.direction);
    float c = dot_product(oc, oc) - sphere.radius * sphere.radius;
    
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    
    t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return t > 0.001f; // Avoid self-intersection
}

__global__ void recursiveRayTrace(Ray *rays, float3 *colors, Sphere *spheres, int numSpheres,
                                 int numRays, int maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    
    Ray ray = rays[idx];
    float3 color = make_float3_device(0.0f, 0.0f, 0.0f);
    
    if (ray.depth >= maxDepth) {
        colors[idx] = color;
        return;
    }
    
    // Find closest intersection
    float closest_t = INFINITY;
    int closest_sphere = -1;
    
    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (intersectSphere(ray, spheres[i], t) && t < closest_t) {
            closest_t = t;
            closest_sphere = i;
        }
    }
    
    if (closest_sphere >= 0) {
        // Hit a sphere - compute reflection ray and launch child kernel
        float3 hitPoint = add_float3(ray.origin, scale_float3(ray.direction, closest_t));
        float3 normal = subtract_float3(hitPoint, spheres[closest_sphere].center);
        normal = scale_float3(normal, 1.0f / spheres[closest_sphere].radius); // normalize
        
        // Reflect direction: r = d - 2(d·n)n
        float dotProduct = dot_product(ray.direction, normal);
        float3 reflection = subtract_float3(ray.direction, scale_float3(normal, 2.0f * dotProduct));
        
        // Create reflection ray
        Ray *reflectedRay;
        float3 *reflectedColor;
        
        if (ray.depth < maxDepth - 1) {
            cudaMalloc(&reflectedRay, sizeof(Ray));
            cudaMalloc(&reflectedColor, sizeof(float3));
            
            Ray newRay;
            newRay.origin = hitPoint;
            newRay.direction = reflection;
            newRay.depth = ray.depth + 1;
            
            cudaMemcpy(reflectedRay, &newRay, sizeof(Ray), cudaMemcpyHostToDevice);
            
            // Launch child kernel for reflection
            recursiveRayTrace<<<1, 1>>>(reflectedRay, reflectedColor, spheres, 
                                       numSpheres, 1, maxDepth);
            
            cudaDeviceSynchronize();
            
            float3 reflColor;
            cudaMemcpy(&reflColor, reflectedColor, sizeof(float3), cudaMemcpyDeviceToHost);
            
            // Combine colors (simplified)
            color = add_float3(scale_float3(spheres[closest_sphere].color, 0.3f),
                              scale_float3(reflColor, 0.7f));
            
            cudaFree(reflectedRay);
            cudaFree(reflectedColor);
        } else {
            color = spheres[closest_sphere].color;
        }
    }
    
    colors[idx] = color;
}

// Device-side quicksort implementation
__device__ void deviceQuicksort(float *data, int left, int right, int depth) {
    if (left >= right || depth <= 0) return;
    
    int pivotIndex = devicePartition(data, left, right);
    
    // Launch child kernels for recursive calls if problem is large enough
    if (right - left > MIN_SIZE && depth > 1) {
        // Launch left partition
        if (pivotIndex - 1 > left) {
            deviceQuicksort<<<1, 1>>>(data, left, pivotIndex - 1, depth - 1);
        }
        
        // Launch right partition  
        if (pivotIndex + 1 < right) {
            deviceQuicksort<<<1, 1>>>(data, pivotIndex + 1, right, depth - 1);
        }
        
        cudaDeviceSynchronize();
    } else {
        // Sequential sort for small arrays
        for (int i = left + 1; i <= right; i++) {
            float key = data[i];
            int j = i - 1;
            while (j >= left && data[j] > key) {
                data[j + 1] = data[j];
                j--;
            }
            data[j + 1] = key;
        }
    }
}

__device__ int devicePartition(float *data, int left, int right) {
    float pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            // Swap data[i] and data[j]
            float temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    // Swap data[i+1] and data[right]
    float temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;
    
    return i + 1;
}

__global__ void dynamicQuicksort(float *data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        deviceQuicksort(data, 0, n - 1, MAX_DEPTH);
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

void checkDynamicParallelismSupport() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("Dynamic Parallelism Support Analysis:\n");
    printf("====================================\n");
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        
        bool supportsDynamicParallelism = (prop.major >= 3 && prop.minor >= 5) || prop.major >= 4;
        printf("  Dynamic Parallelism: %s\n", supportsDynamicParallelism ? "Supported" : "Not Supported");
        
        if (supportsDynamicParallelism) {
            printf("  Max Grid Size: %d x %d x %d\n", 
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
            printf("  Device-side malloc: %s\n", 
                   (prop.major >= 2) ? "Supported" : "Not Supported");
        }
        
        printf("\n");
    }
}

bool verifySorted(float *data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}

void testDynamicQuicksort(int n) {
    printf("Testing Dynamic Quicksort with %d elements...\n", n);
    
    size_t bytes = n * sizeof(float);
    
    // Allocate and initialize host data
    float *h_data = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(rand() % 10000);
    }
    
    printf("First 10 elements before sort: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_data[i]);
    }
    printf("\n");
    
    // Allocate device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Time the dynamic quicksort
    auto start = std::chrono::high_resolution_clock::now();
    
    dynamicQuicksort<<<1, 1>>>(d_data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    
    bool sorted = verifySorted(h_data, n);
    
    printf("Dynamic quicksort time: %.2f ms\n", time);
    printf("Result: %s\n", sorted ? "SORTED" : "NOT SORTED");
    printf("First 10 elements after sort: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_data[i]);
    }
    printf("\n\n");
    
    // Compare with CPU sort
    srand(42);
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(rand() % 10000);
    }
    
    start = std::chrono::high_resolution_clock::now();
    std::sort(h_data, h_data + n);
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("CPU std::sort time: %.2f ms\n", cpu_time);
    printf("GPU speedup: %.2fx\n", cpu_time / time);
    
    free(h_data);
    cudaFree(d_data);
}

void testAdaptiveMeshRefinement() {
    printf("Testing Adaptive Mesh Refinement...\n");
    
    const int width = 64, height = 64;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);
    
    float *h_data = (float*)malloc(bytes);
    bool *h_flags = (bool*)malloc(size * sizeof(bool));
    
    // Initialize with a simple function that has high gradients in some regions
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float fx = (float)x / width;
            float fy = (float)y / height;
            h_data[y * width + x] = sinf(fx * 10.0f) * cosf(fy * 10.0f);
        }
    }
    
    float *d_data;
    bool *d_flags;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_flags, size * sizeof(bool)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Launch adaptive mesh refinement
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    adaptiveMeshRefinement<<<grid, block>>>(d_data, d_flags, width, height, 0, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flags, d_flags, size * sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Count refined cells
    int refinedCells = 0;
    for (int i = 0; i < size; i++) {
        if (h_flags[i]) refinedCells++;
    }
    
    printf("Adaptive mesh refinement completed in %.2f ms\n", time);
    printf("Grid size: %dx%d (%d cells)\n", width, height, size);
    printf("Refined cells: %d (%.1f%%)\n", refinedCells, 
           (refinedCells * 100.0f) / size);
    
    free(h_data);
    free(h_flags);
    cudaFree(d_data);
    cudaFree(d_flags);
    
    printf("\n");
}

void testRecursiveRayTracing() {
    printf("Testing Recursive Ray Tracing...\n");
    
    const int numRays = 1024;
    const int numSpheres = 3;
    const int maxDepth = 4;
    
    // Setup scene
    Sphere h_spheres[numSpheres];
    h_spheres[0] = {{0.0f, 0.0f, -5.0f}, 1.0f, {1.0f, 0.0f, 0.0f}}; // Red sphere
    h_spheres[1] = {{2.0f, 0.0f, -4.0f}, 0.8f, {0.0f, 1.0f, 0.0f}}; // Green sphere
    h_spheres[2] = {{-1.5f, 1.0f, -3.0f}, 0.6f, {0.0f, 0.0f, 1.0f}}; // Blue sphere
    
    // Setup rays
    Ray *h_rays = (Ray*)malloc(numRays * sizeof(Ray));
    float3 *h_colors = (float3*)malloc(numRays * sizeof(float3));
    
    for (int i = 0; i < numRays; i++) {
        float x = (i % 32 - 16) * 0.1f;
        float y = (i / 32 - 16) * 0.1f;
        
        h_rays[i].origin = {0.0f, 0.0f, 0.0f};
        h_rays[i].direction = {x, y, -1.0f}; // Normalize in practice
        h_rays[i].depth = 0;
    }
    
    // Allocate device memory
    Ray *d_rays;
    float3 *d_colors;
    Sphere *d_spheres;
    
    CUDA_CHECK(cudaMalloc(&d_rays, numRays * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_colors, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere)));
    
    CUDA_CHECK(cudaMemcpy(d_rays, h_rays, numRays * sizeof(Ray), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    
    // Launch ray tracing
    dim3 block(256);
    dim3 grid((numRays + block.x - 1) / block.x);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    recursiveRayTrace<<<grid, block>>>(d_rays, d_colors, d_spheres, numSpheres, numRays, maxDepth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_colors, d_colors, numRays * sizeof(float3), cudaMemcpyDeviceToHost));
    
    printf("Recursive ray tracing completed in %.2f ms\n", time);
    printf("Traced %d rays with max depth %d\n", numRays, maxDepth);
    printf("Scene: %d spheres\n", numSpheres);
    
    // Show some sample colors
    printf("Sample colors (first 5 rays):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Ray %d: (%.2f, %.2f, %.2f)\n", i, 
               h_colors[i].x, h_colors[i].y, h_colors[i].z);
    }
    
    free(h_rays);
    free(h_colors);
    cudaFree(d_rays);
    cudaFree(d_colors);
    cudaFree(d_spheres);
    
    printf("\n");
}

int main() {
    printf("CUDA Dynamic Parallelism Demonstration\n");
    printf("======================================\n\n");
    
    checkDynamicParallelismSupport();
    
    // Check if current device supports dynamic parallelism
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    bool supportsDynamicParallelism = (prop.major >= 3 && prop.minor >= 5) || prop.major >= 4;
    
    if (!supportsDynamicParallelism) {
        printf("Current device does not support dynamic parallelism.\n");
        printf("Dynamic parallelism requires compute capability 3.5 or higher.\n");
        return 1;
    }
    
    printf("Running dynamic parallelism examples...\n\n");
    
    // Test 1: Dynamic Quicksort
    printf("=== Test 1: Dynamic Quicksort ===\n");
    testDynamicQuicksort(16384);
    
    // Test 2: Adaptive Mesh Refinement
    printf("=== Test 2: Adaptive Mesh Refinement ===\n");
    testAdaptiveMeshRefinement();
    
    // Test 3: Recursive Ray Tracing
    printf("=== Test 3: Recursive Ray Tracing ===\n");
    testRecursiveRayTracing();
    
    // Performance analysis
    printf("=== Performance Analysis ===\n");
    printf("Key Insights:\n");
    printf("1. Dynamic parallelism adds overhead due to kernel launch costs\n");
    printf("2. Best suited for irregular, data-dependent algorithms\n");
    printf("3. Recursive algorithms benefit from natural expression\n");
    printf("4. Memory allocation/deallocation can be expensive on device\n");
    
    printf("\nBest Practices:\n");
    printf("- Use for algorithms with irregular parallelism\n");
    printf("- Minimize dynamic memory allocation\n");
    printf("- Consider launch overhead vs. computation ratio\n");
    printf("- Test against alternative implementations\n");
    printf("- Use appropriate recursion depth limits\n");
    
    printf("\nApplications:\n");
    printf("- Adaptive algorithms (AMR, adaptive sampling)\n");
    printf("- Tree traversal and graph algorithms\n");
    printf("- Recursive mathematical computations\n");
    printf("- Ray tracing and global illumination\n");
    printf("- Divide-and-conquer algorithms\n");
    
    printf("\nDynamic Parallelism demonstration completed!\n");
    return 0;
}