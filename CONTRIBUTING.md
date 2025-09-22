# Contributing to GPU Programming 101

We welcome contributions to GPU Programming 101! This project aims to provide a comprehensive, high-quality educational resource for learning GPU programming with both CUDA and HIP.

## 🤝 Ways to Contribute

- **Report bugs** or suggest improvements
- **Fix typos** or improve documentation
- **Add new examples** or exercises
- **Improve existing code** for better performance or clarity
- **Add support for new GPU architectures**
- **Create translations** for international users
- **Write tutorials** or additional explanations

## 📋 Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss major changes before implementing
3. **Follow our coding standards** and documentation style
4. **Test your changes** on both CUDA and HIP platforms when applicable

## 🔧 Development Setup

### Prerequisites

- **Docker** (recommended) or native CUDA/ROCm installation
- **Git** for version control
- **Make** for building examples
- **Python 3.8+** for documentation tools

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gpu-programming-101.git
cd gpu-programming-101

# Option 1: Using Docker (Recommended)
cd docker
docker-compose up -d cuda-dev  # For NVIDIA GPUs
docker-compose up -d rocm-dev  # For AMD GPUs

# Option 2: Native development
# Install CUDA Toolkit 13.0.1+ or ROCm latest
# See modules/module1/README.md for detailed setup instructions

# Build all examples
make all

# Test specific module
make -C modules/module1/examples all
```

## 📝 Coding Standards

### General Guidelines

- **Cross-platform compatibility**: Provide both CUDA and HIP implementations
- **Consistent naming**: Use descriptive variable and function names
- **Code comments**: Explain complex algorithms and optimizations
- **Error handling**: Use proper error checking macros
- **Performance**: Optimize for clarity first, then performance

### Code Style

```cpp
// Use consistent indentation (4 spaces)
// Include comprehensive error checking
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Document kernel parameters and behavior
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    // Clear comments explaining the algorithm
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### File Organization

```
modules/moduleX/
├── README.md           # Module overview and learning objectives
├── content.md          # Detailed theoretical content
└── examples/
    ├── Makefile        # Build configuration
    ├── 01_topic_cuda.cu      # CUDA implementation
    ├── 01_topic_hip.cpp      # HIP implementation
    └── README.md       # Example-specific documentation
```

## 🧪 Testing Guidelines

### Before Submitting

1. **Compile all examples** in your modified modules
2. **Test on both platforms** when possible (CUDA and HIP)
3. **Verify documentation** renders correctly
4. **Check for memory leaks** in GPU code
5. **Validate performance** doesn't regress

### Testing Commands

```bash
# Test specific module
make -C modules/moduleX/examples clean all

# Test Docker environments
docker-compose exec cuda-dev make -C modules/moduleX/examples all
docker-compose exec rocm-dev make -C modules/moduleX/examples all

# Run performance benchmarks
./modules/moduleX/examples/benchmark.sh
```

## 📚 Documentation Standards

### README Files

- **Clear learning objectives** for each module
- **Prerequisites** and setup instructions
- **Example descriptions** with expected outcomes
- **Performance notes** and optimization explanations

### Code Comments

- **Header comments** explaining file purpose
- **Kernel documentation** describing algorithm and parameters
- **Inline comments** for complex operations
- **Performance notes** for optimization choices

### Markdown Style

```markdown
# Use clear hierarchy with headers

## Code blocks with language specification
```cpp
// Include syntax highlighting
__global__ void example() {
    // Well-commented code
}
```

## 🔄 Submission Process

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
git clone https://github.com/yourusername/gpu-programming-101.git
cd gpu-programming-101

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow coding standards above
- Add tests for new functionality
- Update documentation as needed
- Ensure backward compatibility

### 3. Test Thoroughly

```bash
# Build and test your changes
make clean all

# Test on multiple modules if changes are broad
for module in modules/module*/examples; do
    make -C "$module" clean all || echo "Failed: $module"
done
```

### 4. Submit Pull Request

1. **Push to your fork**:
   ```bash
   git add .
   git commit -m "feat: descriptive commit message"
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - **Clear title** describing the change
   - **Detailed description** of what was modified
   - **Testing information** showing verification
   - **Screenshots** for visual changes
   - **Breaking changes** clearly noted

## 📋 Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Compiled successfully on CUDA
- [ ] Compiled successfully on HIP  
- [ ] Tested example execution
- [ ] Verified documentation updates
- [ ] No performance regression

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code, particularly complex areas
- [ ] I have made corresponding documentation changes
- [ ] My changes generate no new warnings
```

## 🐛 Bug Reports

When reporting bugs, please include:

### Environment Information
- **Operating System**: (Ubuntu 22.04, Windows 11, etc.)
- **GPU**: (RTX 4090, RX 7900 XTX, etc.)
- **Driver Version**: (NVIDIA 535.x, ROCm latest, etc.)
- **CUDA/HIP Version**: (13.0.1, 7.0.1, etc.)
- **Docker**: (if using containerized development)

### Bug Description
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Steps to reproduce**: Minimal steps to trigger the issue
- **Error messages**: Complete error output
- **Code snippets**: Minimal reproducing example

### Bug Report Template

```markdown
**Environment:**
- OS: Ubuntu 22.04
- GPU: RTX 4080
- CUDA: 13.0.1
- Driver: 535.98

**Description:**
Clear description of the issue.

**Steps to Reproduce:**
1. Navigate to module1/examples
2. Run `make 01_hello_world_cuda`
3. Execute `./01_hello_world_cuda`

**Expected:** Program should print "Hello, World!"
**Actual:** Segmentation fault

**Error Output:**
```
[paste complete error here]
```

**Additional Context:**
Any other relevant information.
```

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** for similar requests
2. **Open a discussion issue** before implementing large features
3. **Provide clear motivation** for the feature
4. **Consider cross-platform impact** (CUDA and HIP)
5. **Think about educational value** for learners

## 🏆 Recognition

Contributors will be recognized in:

- **AUTHORS.md** file with contribution details
- **Release notes** for significant contributions  
- **Module credits** for substantial educational content
- **GitHub contributors** page

## ❓ Questions?

- **Open an issue** with the "question" label
- **Join discussions** in GitHub Discussions
- **Check existing documentation** in module README files
- **Review closed issues** for similar questions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Thank you for contributing to GPU Programming 101! Your efforts help make GPU computing more accessible to developers worldwide. 🚀

## 🧩 Maintaining feature docs

If you update examples or module content to use new CUDA or ROCm capabilities, please also:

- Bump the versions in `CUDA_ROCM_FEATURES.md` and re‑scan the official release notes.
- Update module READMEs to mention any new minimum driver/toolkit requirements.
- Avoid marketing claims; prefer links to vendor docs and measured results in our own benchmarks.