# SAF - Algorithmic Storage System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://github.com/xuyedev/saf)
[![Tests](https://img.shields.io/badge/tests-169%20passed-success.svg)](https://github.com/xuyedev/saf)

[中文文档](README_CN.md) | English

---

## 📖 What is SAF?

**SAF (Scientific Algorithmic Format)** is an innovative storage system designed for scientific computing data. Instead of storing the data itself, SAF stores the **algorithm that can regenerate the data**, achieving compression ratios of **100×~100,000×** while maintaining **bit-level lossless** reconstruction.

### Core Insight

> Why store the computed results when you can store the algorithm that generates them?

**Traditional Compression** (gzip/zstd):
- Theory: Statistical redundancy elimination (Shannon Entropy)
- Compression ratio: 2×~5×
- Limitation: Cannot exploit generative patterns

**Algorithmic Storage** (SAF):
- Theory: Kolmogorov Complexity (shortest program length)
- Compression ratio: **100×~100,000×**
- Advantage: Discovers and stores generative rules

---

## ✨ Key Features

- 🚀 **Ultra-High Compression Ratio**: 100×~100,000× for pattern-rich data
- 🔬 **Automatic Pattern Detection**: Detects mathematical sequences, fractals, procedural patterns
- ✅ **Bit-Level Lossless**: 100% data integrity guaranteed (verified by hash)
- 📊 **Supports Multiple Data Types**: Sequences, fractal images, Perlin noise, 2D patterns
- ⚡ **High Performance**: Parallel processing support, 28% speedup for large files (4 threads)
- 🛠️ **Easy to Use**: CLI commands and Python API

---

## 📊 Performance Metrics

Based on comprehensive testing (20 test cases, 100% completed):

| Metric | Value | Description |
|--------|-------|-------------|
| **Highest Compression Ratio** | **82,123×** | 4K Mandelbrot fractal (31.64 MB → 404 B) |
| **Average Compression Ratio** | **2,195×** | Batch test (10 files) |
| **Fastest Decompression** | **6.7 ms** | Stripe pattern (1.00 MB) |
| **Maximum Data Size** | **31.64 MB** | 4K fractal image (8.3M pixels) |
| **Batch Success Rate** | **100%** | 10/10 files passed |
| **Concurrent Success Rate** | **100%** | 9/9 operations passed |
| **Concurrent Speedup** | **28%** | Large files (4 threads) |

### Example Compression Results

| Data Type | Original Size | Compressed Size | Ratio | Algorithm |
|-----------|--------------|-----------------|-------|-----------|
| Fibonacci (1M terms) | 7.63 MB | 408 B | **19,608×** | fibonacci |
| Mandelbrot (4K) | 31.64 MB | 404 B | **82,123×** | mandelbrot |
| Perlin Noise | 2.00 MB | 407 B | **5,153×** | perlin_noise |
| Checkerboard | 1.00 MB | 363 B | **2,889×** | checkerboard |
| Prime Numbers | 79 KB | 314 B | **254×** | primes |

---

## 🎯 Ideal Use Cases

### ✅ Perfect Fit (Compression Ratio > 100×)

- Mathematical sequences (Fibonacci, primes, polynomials)
- Fractal images (Mandelbrot, Julia sets)
- Procedurally generated content (Perlin noise, checkerboard, stripes)
- Physics simulation results (following known laws)
- Parametric 3D models

### ⚠️ Not Suitable

- Real photos (no deterministic generation algorithm)
- Natural language text (semantic complexity)
- True random data (white noise)
- Very small data (< 1KB, metadata overhead)

---

## 🚧 Project Status

**Current Version**: v0.2.0 (Testing & Optimization Phase)

- ✅ Core functionality complete (7/8 development stages)
- ✅ Comprehensive testing complete (20/20 tests, 169 unit tests)
- ✅ Test coverage: 87%
- ✅ Concurrent processing verified (thread-safe)
- ⚠️ **Note**: This software is under active development. Features may change without notice. Please test thoroughly before production use.

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/xuyedev/saf.git
cd saf

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAF
pip install -e .
```

### Verify Installation

```bash
saf --help
```

---

## 🚀 Quick Start

### Command Line Interface

#### 1. Compress a File

```bash
# Compress a numpy array (.npy file)
saf compress data/sequences/fibonacci_10000.npy -o output.saf

# Output:
# ✓ Compression completed
# Algorithm: fibonacci
# Confidence: 100.0%
# Compression ratio: 198.3×
# Original: 79.00 KB → Compressed: 404 B
```

#### 2. Decompress a File

```bash
# Decompress .saf file
saf decompress output.saf -o restored.npy

# Output:
# ✓ Decompression completed
# Data restored successfully
```

#### 3. Verify Lossless Integrity

```bash
# Verify data integrity
saf verify data/sequences/fibonacci_10000.npy restored.npy

# Output:
# ✓ Verification passed
# Hash match: ✓
# Bit-level lossless: ✓
```

#### 4. Batch Benchmark

```bash
# Run benchmark on directory
saf benchmark data/sequences/

# Output:
# Processing 10 files...
# Average compression ratio: 2195×
# Total time: 33.09s
# Success rate: 100%
```

### Python API

```python
from src.storage.compressor import Compressor
from src.storage.decompressor import Decompressor
import numpy as np

# Load data
data = np.load('data/sequences/fibonacci_10000.npy')

# Compress
compressor = Compressor()
result = compressor.compress(data, 'output.saf')
print(f"Compression ratio: {result.compression_ratio:.1f}×")

# Decompress
decompressor = Decompressor()
restored_data = decompressor.decompress('output.saf')

# Verify
assert np.array_equal(data, restored_data)
print("✓ Bit-level lossless verified!")
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Scientific Computing** | NumPy, SciPy |
| **Symbolic Computation** | SymPy |
| **CLI Framework** | Click |
| **Serialization** | msgpack |
| **Testing** | pytest, pytest-cov |
| **Progress Display** | tqdm |
| **Type Checking** | mypy |
| **Code Formatting** | black, ruff |

---

## 📁 Project Structure

```
saf/
├── src/                      # Source code
│   ├── cli/                  # Command-line interface
│   ├── detectors/            # Pattern detectors
│   ├── generators/           # Data generators
│   ├── storage/              # Compression/decompression engine
│   ├── utils/                # Utility functions
│   └── verification/         # Verification and benchmarking
├── tests/                    # Test suite (169 tests, 87% coverage)
├── data/                     # Sample data
├── config/                   # Configuration files
├── docs/                     # Documentation
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 🧪 Testing & Quality

- **Unit Tests**: 169 tests (100% passed)
- **Test Coverage**: 87%
- **Integration Tests**: End-to-end workflow verified
- **Concurrent Tests**: Thread-safety verified (100% success)
- **Large-Scale Tests**: 4K images (31.64 MB), 1M sequences (7.63 MB)
- **Type Checking**: mypy strict mode
- **Code Style**: PEP 8 compliant (black + ruff)

---

## ⚠️ Known Limitations

### Performance Constraints

- **Python GIL**: Limits CPU-bound concurrent efficiency
- **Small File Overhead**: < 1KB files have high metadata overhead
- **Detection Time**: Pattern detection may take seconds for large data

### Unsupported Data Types

- π digits sequence (computation cost > storage cost)
- True random data (white noise)
- Float64 fractals (requires feature extension)

### Recommended Use Cases

- **Best**: Large-scale batch processing (> 1MB files)
- **Suitable**: Archive storage, bandwidth-limited data transfer
- **Avoid**: Real-time compression, extremely small files

---

## 📚 Documentation

- [User Guide](docs/user_guide.md) - Detailed usage instructions
- [API Reference](docs/api_reference.md) - Python API documentation
- [Developer Guide](docs/developer_guide.md) - Architecture and contribution guide
- [Test Plan](测试进度/TEST_PLAN.md) - Comprehensive test results
- [Development Stages](DEVELOPMENT_STAGES.md) - Project roadmap

---

## 💬 Feedback & Support

This project is currently in the **testing and optimization phase**. Your feedback is valuable!

### Report Issues

- 🐛 **Bug Reports**: Please open an issue on GitHub or contact via email
- 💡 **Feature Requests**: Suggestions for improvement are welcome
- 📖 **Documentation**: Help improve the documentation

### Contact

- **Email**: xu3033866090@gmail.com
- **微信 (WeChat)**: xuyedev
- **GitHub Issues**: [https://github.com/xuyedev/saf/issues](https://github.com/xuyedev/saf/issues)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Important**: This software is under active development. Please read the development status notice in the LICENSE file before use.

---

## 🙏 Acknowledgments

This project is based on the theory of **Kolmogorov Complexity** and inspired by research in:

- **Kolmogorov Complexity**: Li, M., & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications*. Springer.
- **Symbolic Regression**: Koza, J. R. (1992). *Genetic Programming*. MIT Press.
- **Fractal Compression**: Barnsley, M. F., & Hurd, L. P. (1993). *Fractal Image Compression*. AK Peters.

---

## 📊 Project Statistics

```
Development Stages: 7/8 completed
Lines of Code: ~8,000
Test Cases: 169 (100% passed)
Test Coverage: 87%
Supported Algorithms: 11 (fibonacci, primes, polynomial, arithmetic, geometric,
                          mandelbrot, julia, checkerboard, stripes, perlin_noise, gzip)
Maximum Compression Ratio: 82,123×
Average Compression Ratio: 2,195×
```

---

**Author**: 徐野 (Xu Ye)
**Date**: 2025-11-23
**Version**: v0.2.0

---

⭐ **If you find this project useful, please consider giving it a star on GitHub!**
