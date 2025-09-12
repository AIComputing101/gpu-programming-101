# GPU Programming 101 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/ROCm-5.0%2B-red?logo=amd)](https://rocmdocs.amd.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

**A comprehensive, hands-on educational project for mastering GPU programming with CUDA and HIP**

*From beginner fundamentals to production-ready optimization techniques*

**Quick Navigation:** [🚀 Quick Start](#-quick-start) • [📚 Modules](#-modules) • [🐳 Docker Setup](#-docker-development) • [📖 Documentation](SUMMARY.md) • [🤝 Contributing](CONTRIBUTING.md)

---

## 📋 Project Overview

**GPU Programming 101** is a complete educational resource for learning modern GPU programming. This project provides:

- **9 comprehensive modules** covering beginner to expert topics
- **70+ working code examples** in both CUDA and HIP
- **Cross-platform support** for NVIDIA and AMD GPUs  
- **Production-ready development environment** with Docker
- **Professional tooling** including profilers, debuggers, and CI/CD

Perfect for students, researchers, and developers looking to master GPU computing.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Complete Curriculum** | 9 progressive modules from basics to advanced topics |
| 💻 **Cross-Platform** | Full CUDA and HIP support for NVIDIA and AMD GPUs |
| 🐳 **Docker Ready** | Complete containerized development environment |
| 🔧 **Production Quality** | Professional build systems, testing, and profiling |
| 📊 **Performance Focus** | Optimization techniques and benchmarking throughout |
| 🌐 **Community Driven** | Open source with comprehensive contribution guidelines |

## 🚀 Quick Start

### Option 1: Docker (Recommended)
Get started immediately without installing CUDA/ROCm on your host system:

```bash
# Clone the repository
git clone https://github.com/AIComputing101/gpu-programming-101.git
cd gpu-programming-101

# Auto-detect your GPU and start development environment
./docker/scripts/run.sh --auto

# Inside container: verify GPU access and start learning
/workspace/test-gpu.sh
cd modules/module1 && make && ./01_vector_addition_cuda
```

### Option 2: Native Installation
For direct system installation:

```bash
# Prerequisites: CUDA 11.0+ or ROCm 5.0+, GCC 7+, Make

# Clone and build
git clone https://github.com/AIComputing101/gpu-programming-101.git
cd gpu-programming-101

# Verify your setup
make check-system

# Build and run first example
make module1
cd modules/module1/examples
./01_vector_addition_cuda
```

### 🎯 What You'll Learn

**👶 Beginner Track** - Start here if you're new to GPU programming
- GPU architecture fundamentals
- Writing your first CUDA/HIP kernels  
- Memory management between CPU and GPU
- Basic parallel algorithms
- Debugging and profiling basics

**🔥 Intermediate Track** - For developers with some parallel programming experience
- Advanced memory optimization techniques
- Multi-dimensional data processing
- GPU architecture deep dive
- Performance engineering
- Multi-GPU programming

**🚀 Advanced Track** - For experts seeking production-level skills
- Fundamental parallel algorithms (reduction, scan, convolution)
- Advanced algorithmic patterns (sorting, sparse matrices)
- Domain-specific applications (ML, scientific computing)
- Production deployment and optimization
- Next-generation GPU architectures

## 📚 Modules

| Module | Level | Duration | Topics | Examples |
|--------|-------|----------|--------|----------|
| [**Module 1**](modules/module1/) | Beginner | 4-6h | GPU Fundamentals, CUDA/HIP Basics | 13 |
| [**Module 2**](modules/module2/) | Beginner-Intermediate | 6-8h | Multi-Dimensional Data Processing | 10 |
| [**Module 3**](modules/module3/) | Intermediate | 6-8h | GPU Architecture & Execution Models | 12 |
| [**Module 4**](modules/module4/) | Intermediate-Advanced | 8-10h | Advanced GPU Programming | 9 |
| [**Module 5**](modules/module5/) | Advanced | 6-8h | Performance Engineering | 5 |
| [**Module 6**](modules/module6/) | Intermediate-Advanced | 8-10h | Fundamental Parallel Algorithms | 10 |
| [**Module 7**](modules/module7/) | Advanced | 8-10h | Advanced Algorithmic Patterns | 4 |
| [**Module 8**](modules/module8/) | Advanced | 10-12h | Domain-Specific Applications | 4 |
| [**Module 9**](modules/module9/) | Expert | 6-8h | Production GPU Programming | 4 |

**📈 Progressive Learning Path: 70+ Examples • 50+ Hours • Beginner to Expert**

**[📖 View Detailed Curriculum →](SUMMARY.md)**

## 🛠️ Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GTX 1060+ or AMD RX 580+ (4GB+ VRAM recommended)
- **System**: 8GB+ RAM (16GB+ recommended for advanced modules)

### Software Requirements
- **OS**: Linux (recommended), Windows 10/11, or macOS
- **CUDA**: 11.0+ for NVIDIA GPUs
- **ROCm**: 5.0+ for AMD GPUs  
- **Compiler**: GCC 7+, Clang 8+, or MSVC 2019+
- **Docker**: For containerized development (recommended)

### Programming Knowledge
- **C/C++**: Intermediate level (pointers, memory management)
- **Command Line**: Basic terminal/shell usage
- **Math**: Linear algebra basics helpful but not required

## 🐳 Docker Development

Experience the full development environment with zero setup:

```bash
# Build development containers
./docker/scripts/build.sh --all

# Start interactive development
./docker/scripts/run.sh cuda    # For NVIDIA GPUs
./docker/scripts/run.sh rocm    # For AMD GPUs
./docker/scripts/run.sh --auto  # Auto-detect GPU type
```

**Docker Benefits:**
- 🎯 Zero host configuration required
- 🔧 Complete development environment (compilers, debuggers, profilers)
- 🌐 Cross-platform testing (test your code on both CUDA and HIP)
- 📦 Isolated and reproducible builds
- 🧹 Easy cleanup when done

**[📖 Complete Docker Guide →](docker/README.md)**

## 🔧 Build System

### Project-Wide Commands
```bash
make all           # Build all modules
make test          # Run comprehensive tests  
make clean         # Clean all artifacts
make check-system  # Verify GPU setup
make status        # Show module completion status
```

### Module-Specific Commands
```bash
cd modules/module1/examples
make               # Build all examples in module
make test          # Run module tests
make profile       # Performance profiling
make debug         # Debug builds with extra checks
```

## 🚦 Getting Started Guide

### 1. Choose Your Path
- **🐳 Docker**: No setup required, works everywhere → [Docker Guide](docker/README.md)  
- **💻 Native**: Direct system installation → [Installation Guide](#option-2-native-installation)

### 2. Start Learning
```bash
# Begin with Module 1
cd modules/module1
cat README.md        # Read learning objectives  
cd examples && make  # Build examples
./01_vector_addition_cuda  # Run your first GPU program!
```

### 3. Progress Through Modules
- Each module builds on previous concepts
- Complete examples and exercises in order
- Use profiling tools to understand performance
- Experiment with different optimizations

### 4. Advanced Topics
- Modules 6-9 cover production-level techniques
- Focus on algorithms and applications relevant to your domain
- Contribute back with improvements and new examples

## 📊 Performance Expectations

| Module Level | Typical GPU Speedup | Memory Efficiency | Code Quality |
|--------------|-------------------|------------------|--------------|
| **Beginner** | 10-100x | 60-80% | Educational |
| **Intermediate** | 50-500x | 80-95% | Optimized |
| **Advanced** | 100-1000x | 85-95% | Production |
| **Expert** | 500-5000x | 95%+ | Library-Quality |

## 🐛 Troubleshooting

### Common Issues & Solutions

**GPU Not Detected**
```bash
# NVIDIA
nvidia-smi  # Should show your GPU
export PATH=/usr/local/cuda/bin:$PATH

# AMD  
rocm-smi   # Should show your GPU
export HIP_PLATFORM=amd
```

**Compilation Errors**
```bash
# Check CUDA installation
nvcc --version
make check-cuda

# Check HIP installation  
hipcc --version
make check-hip
```

**Docker Issues**
```bash
# Test Docker GPU access
./docker/scripts/test.sh

# Rebuild containers
./docker/scripts/build.sh --clean --all
```

**[📖 Full Troubleshooting Guide →](docs/troubleshooting.md)**

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [**SUMMARY.md**](SUMMARY.md) | Complete curriculum overview and learning paths |
| [**CONTRIBUTING.md**](CONTRIBUTING.md) | How to contribute to the project |
| [**Docker Guide**](docker/README.md) | Complete Docker setup and usage |
| [**Module READMEs**](modules/) | Individual module documentation |

## 🤝 Contributing

We welcome contributions from the community! This project thrives on:

- 📝 **New Examples**: Implementing additional GPU algorithms
- 🐛 **Bug Fixes**: Improving existing code and documentation  
- 📚 **Documentation**: Enhancing explanations and tutorials
- 🔧 **Optimizations**: Performance improvements and best practices
- 🌐 **Platform Support**: Cross-platform compatibility improvements

**[📖 Contributing Guidelines →](CONTRIBUTING.md)** • **[🐛 Report Issues →](../../issues)** • **[💡 Request Features →](../../issues/new?template=feature_request.md)**

## 🏆 Community & Support

- 🌟 **Star this project** if you find it helpful!
- 🐛 **Report bugs** using our [issue templates](../../issues/new/choose)
- 💬 **Join discussions** in [GitHub Discussions](../../discussions)
- 📧 **Get help** from the community and maintainers

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TL;DR**: ✅ Commercial use ✅ Modification ✅ Distribution ✅ Private use

## 📚 Citation

If you use this project in your research, education, or publications, please cite it as:

### BibTeX
```bibtex
@misc{gpu-programming-101,
  title={GPU Programming 101: A Comprehensive Educational Project for CUDA and HIP},
  author={{Stephen Shao}},
  year={2025},
  howpublished={\url{https://github.com/AIComputing101/gpu-programming-101}},
  note={A complete GPU programming educational resource with 70+ production-ready examples covering fundamentals through advanced optimization techniques for NVIDIA CUDA and AMD HIP platforms}
}
```

### IEEE Format
Stephen Shao, "GPU Programming 101: A Comprehensive Educational Project for CUDA and HIP," GitHub, 2025. [Online]. Available: https://github.com/AIComputing101/gpu-programming-101

## 🙏 Acknowledgments

- 🎯 **NVIDIA** and **AMD** for excellent GPU computing ecosystems
- 📚 **GPU computing community** for sharing knowledge and best practices  
- 🏫 **Educational institutions** advancing parallel computing education
- 👥 **Contributors** who make this project better every day

---

**Ready to unlock the power of GPU computing?**

**[🚀 Get Started Now](#-quick-start)** • **[📚 View Curriculum](SUMMARY.md)** • **[🐳 Try Docker](docker/README.md)**

---

**⭐ Star this project • 🍴 Fork and contribute • 📢 Share with others**

*Built with ❤️ for the AI Computing 101*