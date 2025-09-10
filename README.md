# GPU Programming 101 🚀

A comprehensive hands-on course for learning GPU programming with CUDA and HIP, covering fundamental concepts through advanced optimization techniques.

## 🎯 Course Overview

This course provides practical, hands-on experience with GPU programming, covering everything from basic parallel computing concepts to advanced optimization techniques. Each module contains theory, working code examples, and exercises.

## 📚 Course Structure

### Module 1: Foundations of GPU Computing ✅
**Status**: Complete  
**Duration**: 4-6 hours  
**Level**: Beginner  

**Topics Covered**:
- GPU architecture and SIMT execution model
- CUDA and HIP programming fundamentals
- Memory management and data transfers
- Basic parallel execution patterns
- Debugging and optimization basics

**[📁 Go to Module 1](modules/module1/)**

### Module 2: Multi-Dimensional Data Processing ✅
**Status**: Complete  
**Duration**: 6-8 hours  
**Level**: Beginner-Intermediate  

**Topics Covered**:
- Multidimensional grid organization
- Thread mapping to data structures
- Image processing kernels
- Matrix multiplication algorithms
- Advanced memory management

**[📁 Go to Module 2](modules/module2/)**

### Module 3: GPU Architecture and Execution Models ✅
**Status**: Complete  
**Duration**: 6-8 hours  
**Level**: Intermediate  

**Topics Covered**:
- GPU architecture deep dive
- Warp scheduling and SIMD hardware
- Control divergence and optimization
- Resource partitioning and occupancy
- Advanced parallel patterns

**[📁 Go to Module 3](modules/module3/)**

### Module 4: Advanced GPU Programming Techniques ✅
**Status**: Complete  
**Duration**: 8-10 hours  
**Level**: Intermediate-Advanced  

**Topics Covered**:
- Multi-GPU programming and scalability
- Asynchronous execution with streams
- Dynamic parallelism techniques
- Advanced memory optimization
- Cross-platform development strategies

**[📁 Go to Module 4](modules/module4/)**

### Module 5: Performance Engineering and Optimization ✅
**Status**: Complete  
**Duration**: 6-8 hours  
**Level**: Advanced  

**Topics Covered**:
- Performance profiling and analysis
- Memory bandwidth optimization
- Kernel optimization strategies
- Bottleneck identification and resolution
- Production performance engineering

**[📁 Go to Module 5](modules/module5/)**

### Module 6: Fundamental Parallel Algorithms 🚧
**Status**: Planned  
**Duration**: 8-10 hours  
**Level**: Intermediate-Advanced  

**Topics**:
- Convolution and filtering algorithms
- Stencil computations
- Histogram and atomic operations
- Reduction patterns and optimizations
- Prefix sum (scan) algorithms

### Module 7: Advanced Algorithmic Patterns 🚧
**Status**: Planned  
**Duration**: 8-10 hours  
**Level**: Advanced  

**Topics**:
- Merge and sorting algorithms
- Sparse matrix computations
- Graph traversal algorithms
- Dynamic programming on GPU
- Load balancing techniques

### Module 8: Domain-Specific Applications 🚧
**Status**: Planned  
**Duration**: 10-12 hours  
**Level**: Advanced  

**Topics**:
- Deep learning inference kernels
- Scientific computing applications
- Image and signal processing
- Monte Carlo simulations
- Numerical methods optimization

### Module 9: Production GPU Programming 🚧
**Status**: Planned  
**Duration**: 6-8 hours  
**Level**: Expert  

**Topics**:
- Cluster computing with MPI
- Dynamic parallelism patterns
- Performance regression testing
- Cross-platform deployment
- Future GPU architectures

## 🛠️ Prerequisites

### Hardware Requirements
- **NVIDIA GPU**: GeForce GTX 1060 or better, or Tesla/Quadro equivalent
- **OR AMD GPU**: RX 580 or better, with ROCm support
- **Memory**: 8GB+ system RAM, 4GB+ GPU memory recommended

### Software Requirements
- **Operating System**: Linux (recommended), Windows 10/11, or macOS
- **CUDA Toolkit**: 11.0+ for NVIDIA GPUs
- **ROCm**: 4.0+ for AMD GPUs
- **Compiler**: GCC 7+, Clang 8+, or MSVC 2019+
- **Build Tools**: Make, CMake (optional)

### Programming Knowledge
- **C/C++**: Intermediate level (pointers, memory management, basic OOP)
- **Command Line**: Basic terminal/shell usage
- **Math**: Linear algebra basics helpful but not required

## 🚀 Quick Start

### Option 1: Docker (Recommended)
Perfect for getting started without installing CUDA/ROCm on your host system.

```bash
git clone https://github.com/yourusername/gpu-programming-101.git
cd gpu-programming-101

# Test your Docker setup
./docker/scripts/test.sh

# Build development container  
./docker/scripts/build.sh --all

# Auto-detect GPU and start appropriate container
./docker/scripts/run.sh --auto

# Inside container - test your GPU
/workspace/test-gpu.sh

# Start learning!
cd modules/module1 && cat README.md
```

### Option 2: Native Installation

```bash
git clone https://github.com/yourusername/gpu-programming-101.git
cd gpu-programming-101

# Check system requirements
# For NVIDIA systems
nvidia-smi && nvcc --version

# For AMD systems  
rocm-smi && hipcc --version

# Start with Module 1
cd modules/module1
cat README.md  # Read module overview
cd examples
make           # Build examples
./04_device_info_cuda  # Check your GPU
```

### 🐳 Docker Benefits
- **No host setup required**: Complete development environment in containers
- **Multi-platform**: Test both CUDA and HIP code easily  
- **Consistent environment**: Same setup across different systems
- **Integrated tools**: Profilers, debuggers, and Jupyter Lab included
- **Easy cleanup**: Remove containers when done

**[📖 Full Docker Guide](docker/README.md)**

### 4. Follow Learning Path
Each module contains:
- **README.md** - Module overview and learning objectives
- **content.md** - Comprehensive theory and explanations  
- **examples/** - Working code examples with build system
- **exercises/** - Additional practice problems (when available)

## 📁 Project Structure

```
gpu-programming-101/
├── README.md                    # This file
├── SUMMARY.md                   # Detailed curriculum overview
├── Makefile                     # Project-wide build system
└── modules/
    ├── module1/                 # Heterogeneous Data Parallel Computing
    │   ├── README.md           # Module overview
    │   ├── content.md          # Theory and explanations
    │   └── examples/           # Working code examples
    │       ├── Makefile
    │       ├── README.md
    │       └── *.cu, *.cpp     # Source files
    ├── module2/                 # Multidimensional Grids and Data
    │   └── [Coming Soon]
    ├── module3/                 # Compute Architecture and Scheduling  
    │   └── [Coming Soon]
    └── [Additional Modules]
```

## 🎓 Learning Path Recommendations

### For Complete Beginners
1. **Start with Module 1** - Focus on understanding basic concepts
2. **Practice extensively** - Modify examples and experiment
3. **Use debugging tools** - Learn proper error handling
4. **Progress gradually** - Master each concept before moving on

### For Experienced Programmers
1. **Skim Module 1 theory** - Focus on GPU-specific concepts
2. **Run all examples** - Understand performance characteristics
3. **Jump to specific topics** - Use course as reference material
4. **Contribute improvements** - Help expand the course content

### For Researchers/Scientists
1. **Focus on relevant modules** - Skip graphics-specific content
2. **Emphasize performance** - Pay special attention to optimization
3. **Explore libraries** - Learn cuBLAS, cuFFT, Thrust, etc.
4. **Real-world applications** - Adapt examples to your domain

## 🔧 Build System

### Project-wide Build
```bash
# Build all available modules
make all

# Build specific module
make module1

# Clean all builds
make clean

# Run tests
make test
```

### Module-specific Build
```bash
cd modules/module1/examples
make          # Build all examples
make vector_add_cuda  # Build specific example
make test     # Run module tests
```

## 🐛 Troubleshooting

### Common Setup Issues

**"nvcc: command not found"**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**"No CUDA-capable device found"**
- Check `nvidia-smi` shows your GPU
- Verify driver installation
- Ensure GPU is not in exclusive/prohibited mode

**"HIP compilation failed"**
```bash
# For AMD GPUs
export HIP_PLATFORM=amd

# For NVIDIA GPUs with HIP
export HIP_PLATFORM=nvidia
```

### Getting Help
- **Module Issues**: Check module-specific README files
- **Code Problems**: Look at debugging examples in Module 1
- **Performance**: Use profiling tools covered in later modules
- **Community**: Create an issue in the repository for help

## 📖 Additional Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [ROCm Documentation](https://rocmdocs.amd.com/)

### Books
- "CUDA by Example" - Sanders, Kandrot
- "Professional CUDA C Programming" - Cheng, Grossman, McKercher
- "GPU Computing Gems" - NVIDIA Corporation

### Online Resources
- [NVIDIA Developer Zone](https://developer.nvidia.com/)
- [AMD Developer Central](https://developer.amd.com/)
- [GPU Computing Community](https://forums.developer.nvidia.com/)

## 🤝 Contributing

We welcome contributions! Please follow standard open source contribution practices.

### Ways to Contribute
- **Add examples** for existing modules
- **Create new modules** following the established structure
- **Improve documentation** and fix typos
- **Add exercises** and solutions
- **Port examples** between CUDA and HIP
- **Performance optimizations** and benchmarks

## 📝 License

This course is released under an open source license. Feel free to use, modify, and distribute for educational purposes.

## 🏆 Acknowledgments

- Thanks to the CUDA and ROCm development communities
- Inspired by hands-on learning approaches in parallel computing education
- Built with contributions from GPU programming educators and practitioners

---

**Happy GPU Programming!** 🚀⚡️

*Last Updated: September 2025*