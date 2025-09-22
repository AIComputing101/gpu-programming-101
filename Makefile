# GPU Programming 101 - Project-wide Makefile
# Builds all available modules and examples

# Default target
.PHONY: all clean test debug profile help module1 module2 module3 module4 module5 module6 module7 module8 module9

# Build all available modules
all: module1 module2 module3 module4 module5 module6 module7 module8 module9

# Module-specific targets
module1:
	@echo "Building Module 1: Foundations of GPU Computing..."
	@$(MAKE) -C modules/module1/examples

module2:
	@echo "Building Module 2: Multi-Dimensional Data Processing..."
	@$(MAKE) -C modules/module2/examples

module3:
	@echo "Building Module 3: GPU Architecture and Execution Models..."
	@$(MAKE) -C modules/module3/examples

module4:
	@echo "Building Module 4: Advanced GPU Programming Techniques..."
	@$(MAKE) -C modules/module4/examples

module5:
	@echo "Building Module 5: Performance Engineering and Optimization..."
	@$(MAKE) -C modules/module5/examples

module6:
	@echo "Building Module 6: Fundamental Parallel Algorithms..."
	@$(MAKE) -C modules/module6/examples

module7:
	@echo "Building Module 7: Advanced Algorithmic Patterns..."
	@$(MAKE) -C modules/module7/examples

module8:
	@echo "Building Module 8: Domain-Specific Applications..."
	@$(MAKE) -C modules/module8/examples

module9:
	@echo "Building Module 9: Production GPU Programming..."
	@$(MAKE) -C modules/module9/examples

# Test all available modules
test: test-module1 test-module2 test-module3 test-module4 test-module5 test-module6 test-module7 test-module8 test-module9

test-module1:
	@echo "Testing Module 1..."
	@$(MAKE) -C modules/module1/examples test

test-module2:
	@echo "Testing Module 2..."
	@$(MAKE) -C modules/module2/examples test

test-module3:
	@echo "Testing Module 3..."
	@$(MAKE) -C modules/module3/examples test

test-module4:
	@echo "Testing Module 4..."
	@$(MAKE) -C modules/module4/examples test

test-module5:
	@echo "Testing Module 5..."
	@$(MAKE) -C modules/module5/examples test

test-module6:
	@echo "Testing Module 6..."
	@$(MAKE) -C modules/module6/examples test

test-module7:
	@echo "Testing Module 7..."
	@$(MAKE) -C modules/module7/examples test

test-module8:
	@echo "Testing Module 8..."
	@$(MAKE) -C modules/module8/examples test

test-module9:
	@echo "Testing Module 9..."
	@$(MAKE) -C modules/module9/examples test

# Debug builds for all modules
debug: debug-module1 debug-module2 debug-module3 debug-module4 debug-module5 debug-module6 debug-module7 debug-module8 debug-module9

debug-module1:
	@echo "Debug build Module 1..."
	@$(MAKE) -C modules/module1/examples debug

debug-module2:
	@echo "Debug build Module 2..."
	@$(MAKE) -C modules/module2/examples debug

debug-module3:
	@echo "Debug build Module 3..."
	@$(MAKE) -C modules/module3/examples debug

debug-module4:
	@echo "Debug build Module 4..."
	@$(MAKE) -C modules/module4/examples debug

debug-module5:
	@echo "Debug build Module 5..."
	@$(MAKE) -C modules/module5/examples debug

debug-module6:
	@echo "Debug build Module 6..."
	@$(MAKE) -C modules/module6/examples debug

debug-module7:
	@echo "Debug build Module 7..."
	@$(MAKE) -C modules/module7/examples debug

debug-module8:
	@echo "Debug build Module 8..."
	@$(MAKE) -C modules/module8/examples debug

debug-module9:
	@echo "Debug build Module 9..."
	@$(MAKE) -C modules/module9/examples debug

# Profile builds for all modules  
profile: profile-module1 profile-module2 profile-module3 profile-module4 profile-module5 profile-module6 profile-module7 profile-module8 profile-module9

profile-module1:
	@echo "Profile build Module 1..."
	@$(MAKE) -C modules/module1/examples profile

profile-module2:
	@echo "Profile build Module 2..."
	@$(MAKE) -C modules/module2/examples profile

profile-module3:
	@echo "Profile build Module 3..."
	@$(MAKE) -C modules/module3/examples profile

profile-module4:
	@echo "Profile build Module 4..."
	@$(MAKE) -C modules/module4/examples profile

profile-module5:
	@echo "Profile build Module 5..."
	@$(MAKE) -C modules/module5/examples profile

profile-module6:
	@echo "Profile build Module 6..."
	@$(MAKE) -C modules/module6/examples profile

profile-module7:
	@echo "Profile build Module 7..."
	@$(MAKE) -C modules/module7/examples profile

profile-module8:
	@echo "Profile build Module 8..."
	@$(MAKE) -C modules/module8/examples profile

profile-module9:
	@echo "Profile build Module 9..."
	@$(MAKE) -C modules/module9/examples profile

# Clean all builds
clean:
	@echo "Cleaning all modules..."
	@$(MAKE) -C modules/module1/examples clean
	@$(MAKE) -C modules/module2/examples clean
	@$(MAKE) -C modules/module3/examples clean
	@$(MAKE) -C modules/module4/examples clean
	@$(MAKE) -C modules/module5/examples clean
	@$(MAKE) -C modules/module6/examples clean
	@$(MAKE) -C modules/module7/examples clean
	@$(MAKE) -C modules/module8/examples clean
	@$(MAKE) -C modules/module9/examples clean
	@echo "Clean completed."

# Development helpers
check-cuda:
	@echo "Checking CUDA installation..."
	@if command -v nvcc > /dev/null 2>&1; then \
		echo "✓ NVCC found: $$(nvcc --version | grep release)"; \
		if command -v nvidia-smi > /dev/null 2>&1; then \
			echo "✓ NVIDIA GPU detected:"; \
			nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1; \
		else \
			echo "⚠ nvidia-smi not found - GPU may not be available"; \
		fi \
	else \
		echo "✗ NVCC not found - please install CUDA toolkit"; \
	fi

check-hip:
	@echo "Checking HIP installation..."  
	@if command -v hipcc > /dev/null 2>&1; then \
		echo "✓ HIPCC found: $$(hipcc --version | head -1)"; \
		if command -v rocm-smi > /dev/null 2>&1; then \
			echo "✓ ROCm detected"; \
			rocm-smi --showproductname 2>/dev/null | head -5 || true; \
		else \
			echo "ℹ rocm-smi not found - may be using CUDA backend"; \
		fi \
	else \
		echo "✗ HIPCC not found - please install ROCm or HIP"; \
	fi

check-system: check-cuda check-hip
	@echo ""
	@echo "System Check Complete"

# Show project structure
structure:
	@echo "GPU Programming 101 Project Structure:"
	@echo "======================================"
	@find . -type d -name ".git" -prune -o -type d -print | sed 's|[^/]*/|  |g'
	@echo ""
	@echo "Available modules:"
	@ls -1 modules/ | sed 's/^/  - /'

# Show available modules and their status
status:
	@echo "GPU Programming 101 - Module Status"
	@echo "=================================="
	@echo ""
	@echo "✅ Module 1: Foundations of GPU Computing"
	@echo "   Status: Complete with examples and exercises"
	@echo "   Location: modules/module1/"
	@echo ""
	@echo "✅ Module 2: Multi-Dimensional Data Processing"  
	@echo "   Status: Complete with examples and exercises"
	@echo "   Location: modules/module2/"
	@echo ""
	@echo "✅ Module 3: GPU Architecture and Execution Models"
	@echo "   Status: Complete with examples and exercises"
	@echo "   Location: modules/module3/"
	@echo ""
	@echo "✅ Module 4: Advanced GPU Programming Techniques"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module4/"
	@echo ""
	@echo "✅ Module 5: Performance Engineering and Optimization"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module5/"
	@echo ""
	@echo "✅ Module 6: Fundamental Parallel Algorithms"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module6/"
	@echo ""
	@echo "✅ Module 7: Advanced Algorithmic Patterns"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module7/"
	@echo ""
	@echo "✅ Module 8: Domain-Specific Applications"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module8/"
	@echo ""
	@echo "✅ Module 9: Production GPU Programming"
	@echo "   Status: Complete with examples and exercises" 
	@echo "   Location: modules/module9/"
	@echo ""
	@echo "🎉 All 9 modules are complete and ready for use!"
	@echo ""
	@echo "Use 'make help' for build instructions"

# Help target
help:
	@echo "GPU Programming 101 - Build System"
	@echo "================================="
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Build all available modules"
	@echo "  clean        - Clean all build artifacts"
	@echo "  test         - Run all available tests"
	@echo "  debug        - Build all modules with debug flags"
	@echo "  profile      - Build all modules with profiling flags"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Module targets:"
	@echo "  module1      - Build Module 1 examples"
	@echo "  module2      - Build Module 2 examples"
	@echo "  module3      - Build Module 3 examples" 
	@echo "  module4      - Build Module 4 examples"
	@echo "  module5      - Build Module 5 examples"
	@echo "  module6      - Build Module 6 examples"
	@echo "  module7      - Build Module 7 examples"
	@echo "  module8      - Build Module 8 examples"
	@echo "  module9      - Build Module 9 examples"
	@echo ""
	@echo "System checks:"
	@echo "  check-system - Check CUDA and HIP installations"
	@echo "  check-cuda   - Check CUDA installation only"
	@echo "  check-hip    - Check HIP installation only"
	@echo ""
	@echo "Information:"
	@echo "  status       - Show module development status"
	@echo "  structure    - Show project directory structure"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make check-system  # Verify your setup"
	@echo "  2. make module1       # Build Module 1 examples"
	@echo "  3. cd modules/module1/examples && ./04_device_info_cuda"
	@echo ""
	@echo "For module-specific builds:"
	@echo "  cd modules/module1/examples && make"

# Default target message
.DEFAULT_GOAL := help