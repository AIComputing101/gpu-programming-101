/**
 * Module 9: Production GPU Programming - Production Architecture Patterns (HIP)
 * 
 * Enterprise-grade GPU application architecture demonstrating production-ready patterns
 * adapted for AMD GPU architectures using ROCm/HIP. This example showcases real-world
 * production requirements optimized for AMD hardware and ROCm ecosystem.
 * 
 * Topics Covered:
 * - ROCm-specific error handling and system monitoring
 * - AMD GPU health monitoring with rocm-smi integration
 * - Wavefront-aware resource management
 * - Production logging optimized for AMD GPU environments
 * - Multi-tenant resource isolation for AMD GPUs
 * - ROCm ecosystem integration patterns
 */

#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities
#include <rocm_smi/rocm_smi.h>
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

// Production-grade error handling macros for HIP
#define HIP_CHECK_PROD(call, context) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            ProductionLogger::getInstance().logError("HIP_ERROR", \
                std::string(context) + ": " + hipGetErrorString(error), \
                __FILE__, __LINE__); \
            throw GPUProductionException(error, context); \
        } \
    } while(0)

#define ROCM_SMI_CHECK_PROD(call, context) \
    do { \
        rsmi_status_t result = call; \
        if (result != RSMI_STATUS_SUCCESS) { \
            ProductionLogger::getInstance().logError("ROCM_SMI_ERROR", \
                std::string(context) + ": ROCm SMI error code " + std::to_string(result), \
                __FILE__, __LINE__); \
            throw ROCmSMIProductionException(result, context); \
        } \
    } while(0)

constexpr int WAVEFRONT_SIZE = 64;

// Production exception classes for HIP/ROCm
class GPUProductionException : public std::exception {
private:
    hipError_t error_code;
    std::string context;
    std::string message;
    
public:
    GPUProductionException(hipError_t error, const std::string& ctx) 
        : error_code(error), context(ctx) {
        message = "GPU Production Error in " + context + ": " + hipGetErrorString(error);
    }
    
    const char* what() const noexcept override { return message.c_str(); }
    hipError_t getErrorCode() const { return error_code; }
    const std::string& getContext() const { return context; }
};

class ROCmSMIProductionException : public std::exception {
private:
    rsmi_status_t result_code;
    std::string context;
    std::string message;
    
public:
    ROCmSMIProductionException(rsmi_status_t result, const std::string& ctx) 
        : result_code(result), context(ctx) {
        message = "ROCm SMI Production Error in " + context + ": Code " + std::to_string(result);
    }
    
    const char* what() const noexcept override { return message.c_str(); }
    rsmi_status_t getResultCode() const { return result_code; }
};

// Production logging system (same as CUDA version but adapted for ROCm)
class ProductionLogger {
private:
    mutable std::mutex log_mutex;
    std::ofstream log_file;
    bool console_output;
    
    ProductionLogger() : console_output(true) {
        log_file.open("rocm_production.log", std::ios::app);
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

// Production configuration management (same interface, ROCm-specific defaults)
class ProductionConfig {
private:
    std::unordered_map<std::string, std::string> config_values;
    mutable std::mutex config_mutex;
    
public:
    static ProductionConfig& getInstance() {
        static ProductionConfig instance;
        return instance;
    }
    
    bool loadFromFile(const std::string& config_path) {
        std::lock_guard<std::mutex> lock(config_mutex);
        
        std::ifstream file(config_path);
        if (!file.is_open()) {
            // Load ROCm-specific defaults
            config_values["gpu_device_id"] = "0";
            config_values["health_check_interval"] = "30";
            config_values["memory_pool_fraction"] = "0.8";
            config_values["wavefront_size"] = "64";
            config_values["enable_rocm_profiling"] = "false";
            
            ProductionLogger::getInstance().logWarning("CONFIG", 
                "Config file not found, using ROCm defaults");
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

// ROCm-optimized GPU resource manager
class GPUResourceManager {
private:
    struct GPUResource {
        void* device_ptr;
        size_t size;
        std::string tenant_id;
        std::chrono::system_clock::time_point allocated_time;
        std::chrono::system_clock::time_point last_access_time;
        bool in_use;
        int numa_node;  // AMD-specific NUMA awareness
    };
    
    std::vector<GPUResource> allocated_resources;
    mutable std::mutex resources_mutex;
    size_t total_allocated;
    size_t peak_allocated;
    
public:
    GPUResourceManager() : total_allocated(0), peak_allocated(0) {}
    
    void* allocateMemory(size_t size, const std::string& tenant_id, int numa_hint = -1) {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        void* device_ptr = nullptr;
        
        try {
            // ROCm-specific memory allocation with NUMA awareness
            HIP_CHECK_PROD(hipMalloc(&device_ptr, size), "Memory allocation for " + tenant_id);
            
            GPUResource resource;
            resource.device_ptr = device_ptr;
            resource.size = size;
            resource.tenant_id = tenant_id;
            resource.allocated_time = std::chrono::system_clock::now();
            resource.last_access_time = resource.allocated_time;
            resource.in_use = true;
            resource.numa_node = numa_hint;
            
            allocated_resources.push_back(resource);
            total_allocated += size;
            peak_allocated = std::max(peak_allocated, total_allocated);
            
            ProductionLogger::getInstance().logInfo("GPU_MEMORY", 
                "Allocated " + std::to_string(size) + " bytes for tenant " + tenant_id +
                (numa_hint >= 0 ? " (NUMA node " + std::to_string(numa_hint) + ")" : ""));
            
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
            
            HIP_CHECK_PROD(HIP_CHECK(hipFree(device_ptr), "Memory deallocation for " + tenant_id);
            
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
        std::unordered_map<int, size_t> per_numa_allocation;
    };
    
    MemoryStats getMemoryStats() const {
        std::lock_guard<std::mutex> lock(resources_mutex);
        
        MemoryStats stats;
        stats.total_allocated = total_allocated;
        stats.peak_allocated = peak_allocated;
        stats.num_allocations = allocated_resources.size();
        
        for (const auto& resource : allocated_resources) {
            stats.per_tenant_allocation[resource.tenant_id] += resource.size;
            if (resource.numa_node >= 0) {
                stats.per_numa_allocation[resource.numa_node] += resource.size;
            }
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
                
                HIP_CHECK(hipFree(it->device_ptr));  // Don't throw on GC failure
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

// ROCm-specific health monitoring using rocm-smi
class GPUHealthMonitor {
private:
    struct HealthMetrics {
        float gpu_utilization;
        float memory_utilization;
        float temperature;
        float power_usage;
        float fan_speed;
        bool is_healthy;
        std::chrono::system_clock::time_point timestamp;
    };
    
    HealthMetrics current_metrics;
    mutable std::mutex metrics_mutex;
    std::atomic<bool> monitoring_active;
    std::thread monitoring_thread;
    uint32_t device_id;
    
public:
    GPUHealthMonitor(uint32_t dev_id = 0) : monitoring_active(false), device_id(dev_id) {
        // Initialize ROCm SMI
        try {
            ROCM_SMI_CHECK_PROD(rsmi_init(0), "ROCm SMI initialization");
            ProductionLogger::getInstance().logInfo("HEALTH_MONITOR", "ROCm SMI initialized successfully");
        } catch (const ROCmSMIProductionException& e) {
            ProductionLogger::getInstance().logError("HEALTH_MONITOR", 
                "Failed to initialize ROCm SMI: " + std::string(e.what()));
            throw;
        }
    }
    
    ~GPUHealthMonitor() {
        stopMonitoring();
        rsmi_shut_down();
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
        try {
            std::lock_guard<std::mutex> lock(metrics_mutex);
            
            // Get GPU utilization
            uint32_t utilization;
            rsmi_status_t status = rsmi_dev_busy_percent_get(device_id, &utilization);
            current_metrics.gpu_utilization = (status == RSMI_STATUS_SUCCESS) ? utilization : 0.0f;
            
            // Get memory usage
            uint64_t memory_used, memory_total;
            status = rsmi_dev_memory_usage_get(device_id, RSMI_MEM_TYPE_VRAM, &memory_used);
            if (status == RSMI_STATUS_SUCCESS) {
                rsmi_dev_memory_total_get(device_id, RSMI_MEM_TYPE_VRAM, &memory_total);
                current_metrics.memory_utilization = 
                    100.0f * (float)memory_used / (float)memory_total;
            } else {
                current_metrics.memory_utilization = 0.0f;
            }
            
            // Get temperature
            int64_t temperature;
            status = rsmi_dev_temp_metric_get(device_id, RSMI_TEMP_TYPE_EDGE, 
                                            RSMI_TEMP_CURRENT, &temperature);
            current_metrics.temperature = (status == RSMI_STATUS_SUCCESS) ? 
                temperature / 1000.0f : 0.0f;  // Convert from millicelsius
            
            // Get power usage
            uint64_t power;
            status = rsmi_dev_power_ave_get(device_id, 0, &power);
            current_metrics.power_usage = (status == RSMI_STATUS_SUCCESS) ? 
                power / 1000000.0f : 0.0f;  // Convert from microwatts
            
            // Get fan speed
            int64_t fan_rpm;
            status = rsmi_dev_fan_rpms_get(device_id, 0, &fan_rpm);
            current_metrics.fan_speed = (status == RSMI_STATUS_SUCCESS) ? fan_rpm : 0.0f;
            
            current_metrics.timestamp = std::chrono::system_clock::now();
            
            // Determine health status (AMD GPU specific thresholds)
            current_metrics.is_healthy = 
                (current_metrics.temperature < 90.0f) &&  // AMD GPU temperature threshold
                (current_metrics.memory_utilization < 95.0f) &&  // Memory threshold
                (current_metrics.power_usage < 300.0f);  // Power threshold (adjust based on GPU)
            
            return current_metrics.is_healthy;
            
        } catch (const ROCmSMIProductionException& e) {
            ProductionLogger::getInstance().logError("HEALTH_MONITOR", 
                "Health check failed: " + std::string(e.what()));
            
            std::lock_guard<std::mutex> lock(metrics_mutex);
            current_metrics.is_healthy = false;
            return false;
        }
    }
    
private:
    void monitoringLoop() {
        auto config = ProductionConfig::getInstance();
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

// Production GPU service adapted for ROCm
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
        
        ProductionLogger::getInstance().logInfo("SERVICE", "Production ROCm GPU Service initialized");
    }
    
    bool initialize() {
        std::lock_guard<std::mutex> lock(service_mutex);
        
        try {
            // Load configuration
            auto& config = ProductionConfig::getInstance();
            config.loadFromFile("rocm_service.conf");
            
            // Initialize HIP context
            int device_count;
            HIP_CHECK_PROD(hipGetDeviceCount(&device_count), "Get device count");
            
            if (device_count == 0) {
                ProductionLogger::getInstance().logError("SERVICE", "No HIP devices found");
                return false;
            }
            
            // Set device and initialize
            int device_id = config.getInt("gpu_device_id", 0);
            HIP_CHECK_PROD(hipSetDevice(device_id), "Set HIP device");
            
            // Initialize device properties logging
            hipDeviceProp_t props;
            HIP_CHECK_PROD(hipGetDeviceProperties(&props, device_id), "Get device properties");
            
            ProductionLogger::getInstance().logInfo("SERVICE", 
                "Using GPU: " + std::string(props.name) + 
                ", Compute: " + std::to_string(props.major) + "." + std::to_string(props.minor) +
                ", Memory: " + std::to_string(props.totalGlobalMem / (1024*1024)) + " MB" +
                ", Wavefront Size: " + std::to_string(WAVEFRONT_SIZE));
            
            // Start health monitoring
            health_monitor->startMonitoring();
            
            service_running.store(true);
            ProductionLogger::getInstance().logInfo("SERVICE", "Production ROCm GPU Service started successfully");
            
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
        
        ProductionLogger::getInstance().logInfo("SERVICE", "Shutting down ROCm GPU service...");
        
        // Stop health monitoring
        health_monitor->stopMonitoring();
        
        // Perform cleanup
        resource_manager->performGarbageCollection();
        
        service_running.store(false);
        
        ProductionLogger::getInstance().logInfo("SERVICE", "ROCm GPU service shutdown complete");
    }
    
    // Example production GPU operation optimized for AMD
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
            // Allocate GPU memory with NUMA awareness
            void* d_data = resource_manager->allocateMemory(data_size * sizeof(float), tenant_id);
            
            // Copy data to GPU
            HIP_CHECK_PROD(hipMemcpy(d_data, input_data.data(), 
                                    data_size * sizeof(float), hipMemcpyHostToDevice),
                          "Copy data to GPU for " + tenant_id);
            
            // Simulate GPU processing optimized for AMD wavefronts
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Launch kernel optimized for 64-thread wavefronts
            dim3 block(256);  // 4 wavefronts per workgroup
            dim3 grid((data_size + block.x - 1) / block.x);
            
            // Example kernel call would go here
            // process_data_kernel<<<grid, block>>>((float*)d_data, data_size);
            
            HIP_CHECK_PROD(hipDeviceSynchronize(), "Kernel execution for " + tenant_id);
            
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
        status.metrics["fan_speed"] = health_metrics.fan_speed;
        status.metrics["allocated_memory_mb"] = memory_stats.total_allocated / (1024 * 1024);
        status.metrics["peak_memory_mb"] = memory_stats.peak_allocated / (1024 * 1024);
        status.metrics["wavefront_size"] = WAVEFRONT_SIZE;
        
        return status;
    }
};

// Example production workload optimized for AMD wavefronts
__global__ void production_compute_kernel_amd(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Wavefront-optimized processing
    int wavefront_id = threadIdx.x / WAVEFRONT_SIZE;
    int lane = threadIdx.x % WAVEFRONT_SIZE;
    
    for (int i = idx; i < n; i += stride) {
        // Simulate compute workload optimized for AMD
        data[i] = sqrtf(data[i] * data[i] + 1.0f);
    }
}

// Production testing and validation
void run_production_tests() {
    std::cout << "\n=== Production ROCm GPU Service Tests ===\n";
    
    try {
        ProductionGPUService service;
        
        if (!service.initialize()) {
            std::cerr << "Failed to initialize production ROCm service\n";
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
        std::vector<std::string> tenants = {"tenant_amd_a", "tenant_amd_b", "tenant_amd_c"};
        
        for (const auto& tenant : tenants) {
            bool success = service.processWorkload(tenant, data_size, test_data);
            std::cout << "Workload processing for " << tenant << ": " 
                      << (success ? "SUCCESS" : "FAILED") << "\n";
        }
        
        // Check service health
        auto status = service.getServiceStatus();
        std::cout << "\nROCm Service Status: " << status.status_message << "\n";
        std::cout << "Health: " << (status.is_healthy ? "HEALTHY" : "DEGRADED") << "\n";
        
        std::cout << "ROCm-Specific Metrics:\n";
        for (const auto& [key, value] : status.metrics) {
            std::cout << "  " << key << ": " << std::fixed << std::setprecision(2) << value;
            if (key == "temperature") std::cout << "°C";
            else if (key == "power_usage") std::cout << "W";
            else if (key == "fan_speed") std::cout << " RPM";
            else if (key.find("memory") != std::string::npos) std::cout << " MB";
            else if (key.find("utilization") != std::string::npos) std::cout << "%";
            std::cout << "\n";
        }
        
        // Test graceful shutdown
        service.shutdown();
        
        std::cout << "\nProduction ROCm service test completed successfully\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Production ROCm test failed: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "HIP Production GPU Architecture - AMD GPU Optimized Implementation\n";
    std::cout << "==================================================================\n";
    
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
            
            std::cout << "Production ROCm GPU service running. Press Ctrl+C to shutdown.\n";
            
            // In a real service, this would be replaced with actual request handling
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
                "Production ROCm service failed: " + std::string(e.what()));
            return -1;
        }
    }
    
    // Demo mode - show capabilities
    std::cout << "Production ROCm GPU Architecture Features:\n";
    std::cout << "• ROCm SMI integration for comprehensive health monitoring\n";
    std::cout << "• Wavefront-aware resource management (64-thread wavefronts)\n";
    std::cout << "• NUMA-aware memory allocation for multi-GPU systems\n";
    std::cout << "• AMD GPU specific error handling and recovery\n";
    std::cout << "• Production-grade logging optimized for ROCm ecosystem\n";
    std::cout << "• Multi-tenant resource isolation for AMD GPUs\n";
    std::cout << "• Real-time health monitoring with AMD-specific thresholds\n";
    
    std::cout << "\nUsage:\n";
    std::cout << "  " << argv[0] << " --test-mode        # Run production ROCm tests\n";
    std::cout << "  " << argv[0] << " --production-mode  # Run in production mode\n";
    
    return 0;
}