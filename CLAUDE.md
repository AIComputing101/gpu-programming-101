# CLAUDE.md - GPU Programming 101 Repository Guide

This document provides comprehensive guidance for Claude Code instances working on the GPU Programming 101 repository - a hands-on project for learning GPU programming with CUDA and HIP.

## Repository Overview

**GPU Programming 101** is a comprehensive educational repository that teaches GPU programming through practical examples and exercises. The project is structured as 9 progressive modules, all complete, each building upon previous concepts to provide a complete learning experience from basic GPU concepts to advanced production-level programming.

### Project Philosophy
- **Hands-on Learning**: Every concept is demonstrated with working code examples
- **Cross-Platform**: Supports both NVIDIA CUDA and AMD HIP/ROCm platforms
- **Production-Ready**: Examples progress from educational to production-quality implementations
- **Comprehensive Testing**: Each module includes extensive testing and profiling capabilities
- **Docker-First**: Complete containerized development environment for consistency

## Repository Structure

```
gpu-programming-101/
├── README.md                    # Main project overview and quick start
├── SUMMARY.md                   # Detailed curriculum overview
├── CLAUDE.md                    # This file - for Claude Code instances
├── Makefile                     # Project-wide build system
├── docker/                      # Complete Docker development environment
│   ├── README.md               # Docker setup and usage guide
│   ├── docker-compose.yml      # Multi-platform orchestration
│   ├── cuda/Dockerfile         # NVIDIA CUDA development image
│   ├── rocm/Dockerfile         # AMD ROCm development image
│   └── scripts/
│       ├── build.sh           # Container build automation
│       ├── run.sh             # Container execution script
│       └── test.sh            # Docker environment testing
└── modules/                     # Project modules (9 modules, all complete)
    ├── module1/                # ✅ COMPLETE: Foundations of GPU Computing
    │   ├── README.md          # Module overview and objectives
    │   ├── content.md         # Comprehensive theory and explanations
    │   └── examples/          # Working CUDA/HIP code examples
    │       ├── Makefile       # Module build system
    │       ├── README.md      # Examples overview
    │       └── *.cu, *.cpp    # Source files (12+ examples)
    ├── module2/                # ✅ COMPLETE: Multi-Dimensional Data Processing
    │   ├── README.md          # Module overview and objectives
    │   ├── content.md         # Comprehensive theory guide
    │   └── examples/          # Multidimensional grid examples
    │       ├── Makefile       # Advanced build system
    │       └── *.cu, *.cpp    # Matrix, image processing examples
    ├── module3/                # ✅ COMPLETE: GPU Architecture and Execution Models
    │   ├── README.md          # Comprehensive algorithm documentation
    │   ├── content.md         # Detailed parallel patterns guide
    │   └── examples/          # 7 comprehensive algorithm examples
    │       ├── Makefile       # Advanced build system
    │       └── *.cu, *.cpp    # Both CUDA and HIP implementations
    ├── module4/                # ✅ COMPLETE: Advanced GPU Programming Techniques
    │   ├── README.md          # Module overview with performance metrics
    │   ├── content.md         # Advanced techniques guide (25,000+ chars)
    │   └── examples/          # Advanced GPU programming examples
    │       ├── Makefile       # Advanced build system with profiling
    │       ├── 01_cuda_streams_basics.cu
    │       ├── 02_multi_gpu_programming.cu
    │       ├── 03_unified_memory.cu
    │       ├── 04_peer_to_peer_communication.cu
    │       └── 05_dynamic_parallelism.cu
    ├── module5/                # ✅ COMPLETE: Performance Engineering and Optimization
    │   ├── README.md          # Performance optimization overview
    │   ├── content.md         # Comprehensive performance theory (25,000+ chars)
    │   └── examples/          # Performance optimization examples
    │       ├── Makefile       # Advanced build system with profiling
    │       ├── 01_gpu_profiling_cuda.cu
    │       ├── 01_hip_profiling.cpp
    │       ├── 02_memory_optimization_cuda.cu
    │       └── 03_kernel_optimization_cuda.cu
    ├── module6/                # ✅ COMPLETE: Fundamental Parallel Algorithms
    ├── module7/                # ✅ COMPLETE: Advanced Algorithmic Patterns
    ├── module8/                # ✅ COMPLETE: Domain-Specific Applications
    └── module9/                # ✅ COMPLETE: Production GPU Programming
```

## Module Status and Content

### ✅ Module 1: Foundations of GPU Computing (COMPLETE)
- **Status**: Production ready with comprehensive examples
- **Content**: GPU fundamentals, CUDA/HIP basics, memory management, debugging
- **Examples**: 12+ working examples covering vector operations, matrix computations, debugging
- **Testing**: Complete test suite with performance benchmarking
- **Documentation**: Full README with learning objectives and quick start

### ✅ Module 2: Multi-Dimensional Data Processing (COMPLETE)
- **Status**: Production ready with comprehensive examples
- **Content**: Multidimensional grid organization, thread mapping, image processing, matrix algorithms
- **Examples**: Comprehensive examples covering 2D/3D grids, image kernels, matrix operations
- **Testing**: Complete test suite with advanced memory management examples
- **Documentation**: Full theory guide with practical demonstrations

### ✅ Module 3: GPU Architecture and Execution Models (COMPLETE)
- **Status**: Production ready with comprehensive algorithm implementations
- **Content**: Fundamental parallel patterns (reduction, scan, sorting, stencil), cooperative groups
- **Examples**: 7 comprehensive examples covering reduction, scan, sorting, convolution, matrix ops, graph algorithms, cooperative groups
- **Cross-Platform**: Both CUDA and HIP implementations for key algorithms
- **Performance**: Detailed complexity analysis and optimization strategies

### ✅ Module 4: Advanced GPU Programming Techniques (COMPLETE) 
- **Status**: Production ready with advanced examples
- **Content**: Multi-GPU programming, CUDA streams, unified memory, P2P communication, dynamic parallelism
- **Examples**: 5 comprehensive examples (~2000+ lines each) demonstrating advanced techniques
- **Performance**: Detailed benchmarking and optimization guidelines
- **Testing**: Advanced testing with multi-GPU detection and profiling integration

### ✅ Module 5: Performance Engineering and Optimization (COMPLETE)
- **Status**: Production ready with comprehensive optimization examples
- **Content**: Performance profiling, memory optimization, kernel optimization, bottleneck analysis
- **Examples**: 4 comprehensive examples covering profiling tools, memory hierarchy, kernel tuning
- **Cross-Platform**: Both CUDA and HIP profiling implementations
- **Testing**: Advanced build system with profiling, benchmarking, and validation targets

### ✅ Module 6: Fundamental Parallel Algorithms (COMPLETE)
- **Status**: Production ready with comprehensive algorithm implementations
- **Content**: Convolution, stencil, histogram, reduction, prefix sum algorithms
- **Examples**: 10 comprehensive examples covering fundamental parallel patterns
- **Cross-Platform**: Both CUDA and HIP implementations for key algorithms
- **Performance**: Detailed complexity analysis and optimization strategies

### ✅ Module 7: Advanced Algorithmic Patterns (COMPLETE) 
- **Status**: Production ready with advanced algorithm implementations
- **Content**: Merge/sort, sparse matrix, graph traversal, dynamic programming
- **Examples**: 4 comprehensive examples covering complex algorithmic patterns
- **Performance**: Advanced optimization techniques and scaling analysis
- **Cross-Platform**: Both CUDA and HIP implementations

### ✅ Module 8: Domain-Specific Applications (COMPLETE)
- **Status**: Production ready with real-world application examples
- **Content**: Deep learning kernels, scientific computing, image processing, Monte Carlo
- **Examples**: 4 comprehensive examples with domain-specific optimizations
- **Integration**: Library integration (cuBLAS, cuDNN, cuFFT, etc.)
- **Cross-Platform**: Both CUDA and HIP implementations

### ✅ Module 9: Production GPU Programming (COMPLETE)
- **Status**: Production ready with enterprise-level examples
- **Content**: Advanced architectures, error handling, memory management, deployment
- **Examples**: 4 comprehensive examples covering production considerations
- **Scope**: Enterprise-level GPU programming best practices
- **Future**: Next-generation GPU architecture considerations

## Build System Architecture

### Project-Level Build (Root Makefile)
```bash
# Key commands:
make all           # Build all available modules (currently: module1 only in root Makefile)
make test          # Run all module tests
make check-system  # Verify CUDA/HIP installation
make status        # Show module development status
make clean         # Clean all builds
```

**Note**: The root Makefile has been updated to include all 9 complete modules with comprehensive build and test targets. You can use both project-wide builds (`make all`) and module-specific builds as needed.

### Module-Level Build
Each module has its own comprehensive Makefile with:
- Individual example compilation
- Comprehensive testing suites  
- Performance benchmarking
- System information detection
- Profiling tool integration
- Cross-platform support (CUDA/HIP)

### Docker Build System
Complete containerized development environment:
- **CUDA Container**: `nvidia/cuda:12.4-devel-ubuntu22.04` base
- **ROCm Container**: `rocm/dev-ubuntu-22.04:6.0` base
- **Features**: GPU drivers, development tools, profilers, Jupyter Lab
- **Scripts**: Automated build, run, and testing scripts

## Development Workflows

### 1. Native Development
```bash
# Quick start
git clone [repository]
cd gpu-programming-101
make check-system     # Verify GPU setup
make module1          # Build Module 1
make module4          # Build Module 4 (Advanced)
make test            # Run comprehensive tests

# Module-specific development
cd modules/module1/examples
make                 # Build all examples
make test           # Run module tests
make system_info    # Check GPU capabilities
./04_device_info_cuda  # Test GPU access
```

### 2. Docker Development (Recommended)
```bash
# Setup (one-time)
./docker/scripts/build.sh --all      # Build both CUDA and ROCm containers
./docker/scripts/test.sh             # Test Docker GPU access

# Development workflow
./docker/scripts/run.sh --auto       # Auto-detect GPU, start container
# Inside container:
/workspace/test-gpu.sh               # Verify GPU access
cd modules/module1/examples && make test
cd modules/module3/examples && make test
cd modules/module4/examples && make test

# Alternative: Jupyter development
./docker/scripts/run.sh cuda --jupyter  # Start with Jupyter Lab
# Access: http://localhost:8888
```

### 3. Multi-Platform Testing
```bash
# Test CUDA implementation
./docker/scripts/run.sh cuda
make module1 && cd modules/module1/examples && make test

# Test HIP implementation (cross-platform)
./docker/scripts/run.sh rocm  
cd modules/module1/examples && make vector_add_hip && ./vector_add_hip
```

## Key Technical Concepts

### Module 1 Fundamentals
- **GPU Architecture**: SIMT execution model, memory hierarchy
- **CUDA Programming**: Kernels, threads, blocks, grids
- **Memory Management**: Host-device transfers, memory types
- **Error Handling**: Comprehensive CUDA error checking patterns
- **Performance**: Basic optimization and profiling

### Module 3 Algorithm Patterns
- **Parallel Algorithms**: Reduction, scan/prefix sum, sorting algorithms
- **Stencil Operations**: Convolution, halo region management, boundary conditions
- **Graph Algorithms**: BFS, shortest path, connected components, PageRank
- **Cooperative Groups**: Modern CUDA programming model, warp-level primitives
- **Memory Optimization**: Shared memory tiling, bank conflict avoidance
- **Cross-Platform**: Both CUDA and HIP implementations

### Module 4 Advanced Topics
- **CUDA Streams**: Asynchronous execution, pipeline processing
- **Multi-GPU Programming**: Load balancing, scaling analysis, NUMA awareness  
- **Unified Memory**: Automatic migration, page fault optimization, prefetching
- **P2P Communication**: Direct GPU-to-GPU transfers, topology optimization
- **Dynamic Parallelism**: GPU-launched kernels, recursive algorithms
- **Profiling**: Nsight Compute, Nsight Systems, performance analysis

### Cross-Platform Support
- **CUDA**: NVIDIA GPU programming model
- **HIP**: AMD's C++ GPU programming interface (also supports NVIDIA)
- **Build Flexibility**: Examples can be compiled for both platforms
- **Testing**: Comprehensive cross-platform validation

## Testing and Quality Assurance

### Automated Testing
- **Compilation Tests**: Verify all examples build successfully
- **Runtime Tests**: Execute examples and validate outputs
- **Performance Benchmarks**: Measure and compare GPU performance
- **Multi-GPU Tests**: Detect and test multiple GPU configurations
- **Cross-Platform**: Test both CUDA and HIP implementations

### Profiling Integration
- **NVIDIA Tools**: Nsight Compute, Nsight Systems integration
- **AMD Tools**: ROCm profiler (rocprof) integration
- **Custom Metrics**: Performance analysis helpers in Makefiles
- **Memory Analysis**: CUDA-memcheck and memory debugging tools

## Common Development Tasks

### Adding New Examples
1. **Location**: `modules/moduleX/examples/`
2. **Naming**: Use descriptive names with numbering (e.g., `06_new_technique.cu`)
3. **Documentation**: Include comprehensive comments and performance notes
4. **Makefile**: Add build rules and test targets
5. **Cross-Platform**: Consider HIP compatibility when possible

### Extending Modules
1. **Content**: Update `content.md` with new theory and explanations
2. **README**: Update module README with new learning objectives
3. **Examples**: Add practical demonstrations of new concepts
4. **Testing**: Extend test suites to cover new functionality
5. **Documentation**: Update quick start and usage examples

### Performance Optimization
1. **Profiling**: Use integrated profiling tools and commands
2. **Benchmarking**: Leverage existing benchmark infrastructure
3. **Analysis**: Document performance characteristics and trade-offs
4. **Comparison**: Include synchronous vs asynchronous comparisons
5. **Scaling**: Test multi-GPU scaling where applicable

## File Organization Standards

### Source Code
- **CUDA Files**: `.cu` extension for CUDA kernels and host code
- **C++ Files**: `.cpp` for host-only code and HIP implementations  
- **Headers**: Minimal use, prefer self-contained examples
- **Comments**: Comprehensive educational comments explaining GPU concepts

### Documentation
- **README.md**: Module overviews, quick start, learning objectives
- **content.md**: Detailed theory, explanations, and best practices
- **Makefile Help**: Each Makefile includes comprehensive help target
- **Performance Notes**: Document expected speedups and characteristics

### Testing
- **Test Targets**: Comprehensive test coverage in all Makefiles
- **System Detection**: Automatic GPU capability detection
- **Cross-Platform**: Graceful fallback for different GPU vendors
- **CI-Ready**: Tests designed for automated execution

## Performance Expectations

### Module 1 (Basic GPU Programming)
- **Vector Addition**: 10-100x speedup over CPU (depending on size)
- **Matrix Operations**: 5-50x speedup with proper memory management
- **Memory Transfers**: Focus on minimizing host-device communication
- **First GPU Program**: Educational focus, performance awareness

### Module 4 (Advanced GPU Programming)
- **CUDA Streams**: 2-4x throughput improvement over synchronous execution
- **Multi-GPU**: Near-linear scaling up to 4 GPUs for appropriate workloads
- **P2P Communication**: 2-10x bandwidth improvement over host-routed transfers
- **Unified Memory**: Simplified programming with 10-30% performance overhead
- **Dynamic Parallelism**: Variable performance, depends on algorithmic fit

## Troubleshooting Common Issues

### Build Problems
- **NVCC Not Found**: Check CUDA installation, PATH configuration
- **Compute Capability**: Ensure GPU supports required compute capability (3.5+ for dynamic parallelism)
- **OpenMP**: Some advanced examples require OpenMP support
- **Docker**: Use containerized environment for consistent builds

### Runtime Issues
- **No GPU Detected**: Check drivers, nvidia-smi/rocm-smi output
- **Memory Errors**: Review CUDA error handling examples in Module 1
- **Performance**: Use profiling tools and performance analysis helpers
- **Multi-GPU**: Verify P2P capabilities, system topology

### Docker Issues
- **GPU Access**: Ensure nvidia-container-toolkit (NVIDIA) or proper device access (AMD)
- **Permissions**: Check Docker group membership and script permissions
- **Port Conflicts**: Use different ports if 8888/8889 are occupied
- **Build Failures**: Clean Docker cache and update base images

## Code Quality Standards

### Educational Code
- **Comprehensive Comments**: Explain GPU programming concepts thoroughly
- **Error Handling**: Demonstrate proper CUDA error checking patterns
- **Performance Notes**: Document expected behavior and optimizations
- **Progressive Complexity**: Start simple, build to production-ready code

### Production Code  
- **Robust Error Handling**: Complete error checking and graceful degradation
- **Performance Optimization**: Apply best practices for memory, computation
- **Scalability**: Design for multiple GPUs and large datasets
- **Cross-Platform**: Consider HIP compatibility for broader adoption

### Testing Code
- **Comprehensive Coverage**: Test both functionality and performance
- **Hardware Detection**: Adapt tests to available GPU capabilities
- **Automated Execution**: Design for CI/CD pipeline integration
- **Clear Output**: Provide clear success/failure indications

## Integration with External Tools

### Profiling Tools
- **NVIDIA Nsight Compute**: Detailed kernel analysis and optimization guidance
- **NVIDIA Nsight Systems**: Timeline analysis and bottleneck identification
- **ROCm Profiler**: AMD GPU performance analysis and optimization
- **Custom Metrics**: Repository includes profiling command examples

### Development Tools
- **Jupyter Lab**: Interactive development environment in Docker containers
- **GPU Debuggers**: cuda-gdb, rocgdb integration examples
- **Memory Tools**: cuda-memcheck, race detection, memory analysis
- **System Monitoring**: nvidia-smi, rocm-smi integration for system analysis

### Build Tools
- **Make**: Primary build system with comprehensive targets
- **Docker**: Complete containerized development environment
- **Docker Compose**: Multi-platform orchestration and service management
- **CMake**: Potential future build system (not currently implemented)

## Repository Maintenance

### Keep Updated
- **CUDA/ROCm Versions**: Update Docker base images regularly
- **Driver Compatibility**: Test with latest GPU drivers
- **Performance Baselines**: Update expected performance metrics
- **Documentation**: Keep README files and guides current

### Quality Assurance  
- **Regular Testing**: Run full test suites across different GPU configurations
- **Performance Regression**: Monitor for performance degradation in updates
- **Cross-Platform**: Validate both CUDA and HIP functionality
- **Docker Health**: Verify container builds and GPU access regularly

### Future Enhancements
- **Additional Modules**: Complete Modules 2 and 3 following established patterns
- **More Examples**: Expand example coverage within existing modules
- **Advanced Techniques**: Add cutting-edge GPU programming techniques
- **Performance Optimization**: Deeper optimization examples and analysis

## Best Practices for Claude Code Instances

### When Working on This Repository
1. **Check Module Status**: Verify which modules are complete vs planned
2. **Use Docker First**: Prefer containerized development for consistency
3. **Test Thoroughly**: Run comprehensive tests after any changes
4. **Follow Patterns**: Maintain consistency with existing code structure
5. **Document Changes**: Update relevant README and documentation files
6. **Cross-Platform Awareness**: Consider CUDA/HIP compatibility

### Performance Considerations
1. **Profile Before Optimizing**: Use integrated profiling tools
2. **Document Performance**: Include expected speedups and characteristics  
3. **Test Scaling**: Verify multi-GPU performance where applicable
4. **Memory Optimization**: Focus on memory bandwidth and access patterns
5. **Iterative Improvement**: Build from educational to production-ready

### Common Commands Reference
```bash
# System verification
make check-system
nvidia-smi          # (NVIDIA GPUs)
rocm-smi           # (AMD GPUs)

# Quick development
make module1 && cd modules/module1/examples && make test
cd modules/module2/examples && make test  # Note: not in root Makefile
cd modules/module3/examples && make test  # Note: not in root Makefile
cd modules/module4/examples && make test  # Note: not in root Makefile
cd modules/module5/examples && make test  # Note: not in root Makefile

# Docker development  
./docker/scripts/build.sh --all
./docker/scripts/run.sh --auto

# Profiling examples
cd modules/module4/examples
make profile_examples    # Show profiling commands
make analyze_memory      # Memory analysis helpers
```

---

**Note**: This repository represents production-quality educational content for GPU programming. Maintain the high standards of documentation, testing, and cross-platform compatibility when making any modifications.