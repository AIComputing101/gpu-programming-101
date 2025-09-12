#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <chrono>

#define MAX_THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Graph representation using compressed sparse row (CSR)
struct CSRGraph {
    int *row_ptr;       // Row pointers
    int *col_indices;   // Column indices  
    float *values;      // Edge weights (optional)
    int num_vertices;
    int num_edges;
};

// Breadth-First Search using frontier-based approach
__global__ void bfs_kernel(int *row_ptr, int *col_indices, int *distances, 
                          bool *frontier, bool *visited, bool *next_frontier, 
                          int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && frontier[tid]) {
        // Process current frontier vertex
        for (int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++) {
            int neighbor = col_indices[edge];
            
            if (!visited[neighbor]) {
                if (atomicCAS(&distances[neighbor], -1, level + 1) == -1) {
                    next_frontier[neighbor] = true;
                }
            }
        }
        frontier[tid] = false;
    }
}

// Single-Source Shortest Path (Bellman-Ford style)
__global__ void sssp_kernel(int *row_ptr, int *col_indices, float *edge_weights,
                           float *distances, bool *updated, int num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        float current_dist = distances[tid];
        if (current_dist != INFINITY) {
            // Relax all outgoing edges
            for (int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++) {
                int neighbor = col_indices[edge];
                float new_dist = current_dist + edge_weights[edge];
                
                if (new_dist < distances[neighbor]) {
                    atomicMinFloat(&distances[neighbor], new_dist);
                    *updated = true;
                }
            }
        }
    }
}

// Custom atomicMin for float (if not available)
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// PageRank algorithm using power iteration
__global__ void pagerank_kernel(int *row_ptr, int *col_indices, int *out_degrees,
                               float *current_pr, float *next_pr, float damping,
                               int num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        float sum = 0.0f;
        
        // Sum contributions from incoming edges
        for (int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++) {
            int neighbor = col_indices[edge];
            if (out_degrees[neighbor] > 0) {
                sum += current_pr[neighbor] / out_degrees[neighbor];
            }
        }
        
        next_pr[tid] = (1.0f - damping) / num_vertices + damping * sum;
    }
}

// Connected Components using label propagation
__global__ void connected_components_kernel(int *row_ptr, int *col_indices,
                                          int *labels, bool *changed, int num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        int current_label = labels[tid];
        int min_label = current_label;
        
        // Find minimum label among neighbors
        for (int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++) {
            int neighbor = col_indices[edge];
            min_label = min(min_label, labels[neighbor]);
        }
        
        // Update label if necessary
        if (min_label < current_label) {
            labels[tid] = min_label;
            *changed = true;
        }
    }
}

// Triangle counting kernel
__global__ void triangle_count_kernel(int *row_ptr, int *col_indices,
                                     long long *triangle_count, int num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long local_count = 0;
    
    if (tid < num_vertices) {
        // For each edge (u,v) from vertex tid
        for (int edge_u = row_ptr[tid]; edge_u < row_ptr[tid + 1]; edge_u++) {
            int v = col_indices[edge_u];
            if (v > tid) { // Avoid double counting
                // Check for common neighbors of tid and v
                int ptr_u = row_ptr[tid];
                int ptr_v = row_ptr[v];
                
                while (ptr_u < row_ptr[tid + 1] && ptr_v < row_ptr[v + 1]) {
                    int neighbor_u = col_indices[ptr_u];
                    int neighbor_v = col_indices[ptr_v];
                    
                    if (neighbor_u == neighbor_v && neighbor_u > v) {
                        local_count++;
                    }
                    
                    if (neighbor_u < neighbor_v) ptr_u++;
                    else ptr_v++;
                }
            }
        }
    }
    
    // Reduce triangle counts across threads
    __shared__ long long shared_count[MAX_THREADS_PER_BLOCK];
    shared_count[threadIdx.x] = local_count;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_count[threadIdx.x] += shared_count[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(triangle_count, shared_count[0]);
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

// Create a simple test graph (grid graph)
CSRGraph createTestGraph(int width, int height) {
    CSRGraph graph;
    graph.num_vertices = width * height;
    
    // Count edges first
    int edge_count = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j < width - 1) edge_count++;  // Right
            if (i < height - 1) edge_count++; // Down
        }
    }
    graph.num_edges = edge_count;
    
    // Allocate memory
    graph.row_ptr = (int*)malloc((graph.num_vertices + 1) * sizeof(int));
    graph.col_indices = (int*)malloc(edge_count * sizeof(int));
    graph.values = (float*)malloc(edge_count * sizeof(float));
    
    // Build CSR representation
    int edge_idx = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int vertex = i * width + j;
            graph.row_ptr[vertex] = edge_idx;
            
            // Right neighbor
            if (j < width - 1) {
                graph.col_indices[edge_idx] = vertex + 1;
                graph.values[edge_idx] = 1.0f;
                edge_idx++;
            }
            
            // Down neighbor  
            if (i < height - 1) {
                graph.col_indices[edge_idx] = vertex + width;
                graph.values[edge_idx] = 1.0f;
                edge_idx++;
            }
        }
    }
    graph.row_ptr[graph.num_vertices] = edge_count;
    
    return graph;
}

int main() {
    printf("CUDA Graph Algorithms Demonstration\n");
    printf("===================================\n\n");
    
    // Create test graph (10x10 grid)
    const int width = 10, height = 10;
    CSRGraph graph = createTestGraph(width, height);
    
    printf("Test Graph Properties:\n");
    printf("  Vertices: %d\n", graph.num_vertices);
    printf("  Edges: %d\n", graph.num_edges);
    printf("  Graph type: %dx%d grid\n\n", width, height);
    
    // Allocate device memory for graph
    int *d_row_ptr, *d_col_indices;
    float *d_values;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (graph.num_vertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, graph.num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, graph.num_edges * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr, 
                         (graph.num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, graph.col_indices, 
                         graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, graph.values, 
                         graph.num_edges * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = MAX_THREADS_PER_BLOCK;
    int blocks = (graph.num_vertices + threads - 1) / threads;
    
    // 1. Breadth-First Search
    printf("1. Breadth-First Search from vertex 0:\n");
    
    int *h_distances = (int*)malloc(graph.num_vertices * sizeof(int));
    bool *h_frontier = (bool*)calloc(graph.num_vertices, sizeof(bool));
    bool *h_visited = (bool*)calloc(graph.num_vertices, sizeof(bool));
    
    // Initialize BFS
    for (int i = 0; i < graph.num_vertices; i++) h_distances[i] = -1;
    h_distances[0] = 0;
    h_frontier[0] = true;
    h_visited[0] = true;
    
    int *d_distances;
    bool *d_frontier, *d_next_frontier, *d_visited;
    CUDA_CHECK(cudaMalloc(&d_distances, graph.num_vertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_visited, graph.num_vertices * sizeof(bool)));
    
    CUDA_CHECK(cudaMemcpy(d_distances, h_distances, graph.num_vertices * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, graph.num_vertices * sizeof(bool), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited, graph.num_vertices * sizeof(bool), 
                         cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int level = 0;
    while (true) {
        CUDA_CHECK(cudaMemset(d_next_frontier, false, graph.num_vertices * sizeof(bool)));
        
        bfs_kernel<<<blocks, threads>>>(d_row_ptr, d_col_indices, d_distances,
                                       d_frontier, d_visited, d_next_frontier,
                                       graph.num_vertices, level);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check if any vertices were added to next frontier
        CUDA_CHECK(cudaMemcpy(h_frontier, d_next_frontier, graph.num_vertices * sizeof(bool),
                             cudaMemcpyDeviceToHost));
        
        bool has_work = false;
        for (int i = 0; i < graph.num_vertices; i++) {
            if (h_frontier[i]) {
                has_work = true;
                h_visited[i] = true;
            }
        }
        
        if (!has_work) break;
        
        CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, graph.num_vertices * sizeof(bool),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_visited, h_visited, graph.num_vertices * sizeof(bool),
                             cudaMemcpyHostToDevice));
        level++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double bfs_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, graph.num_vertices * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("   BFS completed in %.3f ms\n", bfs_time);
    printf("   Maximum distance: %d\n", level);
    printf("   Sample distances: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_distances[i]);
    }
    printf("\n\n");
    
    // 2. Triangle Counting
    printf("2. Triangle Counting:\n");
    
    long long *d_triangle_count;
    long long h_triangle_count = 0;
    CUDA_CHECK(cudaMalloc(&d_triangle_count, sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_triangle_count, &h_triangle_count, sizeof(long long),
                         cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    triangle_count_kernel<<<blocks, threads>>>(d_row_ptr, d_col_indices,
                                              d_triangle_count, graph.num_vertices);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double triangle_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(&h_triangle_count, d_triangle_count, sizeof(long long),
                         cudaMemcpyDeviceToHost));
    
    printf("   Triangle counting completed in %.3f ms\n", triangle_time);
    printf("   Number of triangles found: %lld\n\n", h_triangle_count);
    
    // 3. Connected Components
    printf("3. Connected Components:\n");
    
    int *h_labels = (int*)malloc(graph.num_vertices * sizeof(int));
    for (int i = 0; i < graph.num_vertices; i++) h_labels[i] = i;
    
    int *d_labels;
    bool *d_changed, h_changed;
    CUDA_CHECK(cudaMalloc(&d_labels, graph.num_vertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(bool)));
    
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, graph.num_vertices * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    
    int iterations = 0;
    do {
        h_changed = false;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
        
        connected_components_kernel<<<blocks, threads>>>(d_row_ptr, d_col_indices,
                                                        d_labels, d_changed, graph.num_vertices);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        iterations++;
    } while (h_changed);
    
    end = std::chrono::high_resolution_clock::now();
    double cc_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(h_labels, d_labels, graph.num_vertices * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    // Count unique components
    bool *seen = (bool*)calloc(graph.num_vertices, sizeof(bool));
    int num_components = 0;
    for (int i = 0; i < graph.num_vertices; i++) {
        if (!seen[h_labels[i]]) {
            seen[h_labels[i]] = true;
            num_components++;
        }
    }
    
    printf("   Connected components completed in %.3f ms\n", cc_time);
    printf("   Iterations: %d\n", iterations);
    printf("   Number of components: %d\n\n", num_components);
    
    // Performance Summary
    printf("Performance Summary:\n");
    printf("  Graph: %d vertices, %d edges\n", graph.num_vertices, graph.num_edges);
    printf("  BFS time: %.3f ms\n", bfs_time);
    printf("  Triangle counting time: %.3f ms\n", triangle_time);
    printf("  Connected components time: %.3f ms\n", cc_time);
    
    // Cleanup
    free(graph.row_ptr); free(graph.col_indices); free(graph.values);
    free(h_distances); free(h_frontier); free(h_visited);
    free(h_labels); free(seen);
    
    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values);
    cudaFree(d_distances); cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_visited);
    cudaFree(d_triangle_count); cudaFree(d_labels); cudaFree(d_changed);
    
    printf("\nGraph algorithms demonstration completed!\n");
    return 0;
}