# SAF - 算法式存储系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![测试覆盖率](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://github.com/xuyedev/saf)
[![测试通过](https://img.shields.io/badge/tests-169%20passed-success.svg)](https://github.com/xuyedev/saf)

中文文档 | [English](README.md)

---

## 📖 什么是 SAF？

**SAF (Scientific Algorithmic Format，科学算法格式)** 是一个专为科学计算数据设计的创新存储系统。SAF 不存储数据本身，而是存储**能够重新生成该数据的算法**，从而实现 **100×~100,000×** 的超高压缩比，同时保证 **bit-level 无损还原**。

### 核心洞察

> 为什么要存储计算结果，而不是存储"生成它的算法"？

**传统压缩**（gzip/zstd）：
- 原理：统计冗余消除（Shannon 熵）
- 压缩比：2×~5×
- 局限：无法利用数据的生成规律

**算法式存储**（SAF）：
- 原理：Kolmogorov 复杂度（最短程序长度）
- 压缩比：**100×~100,000×**
- 优势：发现并存储生成规律

---

## ✨ 核心特性

- 🚀 **超高压缩比**：规律性强的数据可达 100×~100,000× 压缩比
- 🔬 **自动模式检测**：自动识别数学序列、分形、程序化图案
- ✅ **bit-level 无损**：100% 数据完整性保证（哈希验证）
- 📊 **支持多种数据类型**：序列、分形图像、Perlin 噪声、2D 图案
- ⚡ **高性能**：支持并发处理，大文件场景性能提升 28%（4线程）
- 🛠️ **易于使用**：提供命令行工具和 Python API

---

## 📊 性能指标

基于全面测试（20 项测试，100% 完成）：

| 指标 | 数值 | 说明 |
|------|------|------|
| **最高压缩比** | **82,123×** | 4K Mandelbrot 分形（31.64 MB → 404 B）|
| **平均压缩比** | **2,195×** | 批量测试（10 个文件）|
| **最快解压速度** | **6.7 ms** | 条纹图案（1.00 MB）|
| **最大数据规模** | **31.64 MB** | 4K 分形图像（830 万像素）|
| **批量成功率** | **100%** | 10/10 文件全部通过 |
| **并发成功率** | **100%** | 9/9 操作全部成功 |
| **并发性能提升** | **28%** | 大文件场景（4 线程）|

### 压缩效果示例

| 数据类型 | 原始大小 | 压缩后 | 压缩比 | 算法 |
|---------|---------|--------|--------|------|
| 斐波那契（100万项）| 7.63 MB | 408 B | **19,608×** | fibonacci |
| Mandelbrot（4K）| 31.64 MB | 404 B | **82,123×** | mandelbrot |
| Perlin 噪声 | 2.00 MB | 407 B | **5,153×** | perlin_noise |
| 棋盘图案 | 1.00 MB | 363 B | **2,889×** | checkerboard |
| 素数序列 | 79 KB | 314 B | **254×** | primes |

---

## 🎯 适用场景

### ✅ 完美适用（压缩比 > 100×）

- 数学序列（斐波那契、素数、多项式）
- 分形图像（Mandelbrot、Julia 集合）
- 程序化生成内容（Perlin 噪声、棋盘、条纹）
- 物理模拟结果（符合已知物理定律）
- 参数化 3D 模型

### ⚠️ 不适用

- 真实照片（无确定性生成算法）
- 自然语言文本（语义复杂）
- 真随机数据（白噪声）
- 极小数据（< 1KB，元数据开销大）

---

## 🚧 项目状态

**当前版本**：v0.2.0（测试与优化阶段）

- ✅ 核心功能已完成（7/8 个开发阶段）
- ✅ 全面测试已完成（20/20 项测试，169 个单元测试）
- ✅ 测试覆盖率：87%
- ✅ 并发处理已验证（线程安全）
- ⚠️ **注意**：本软件正在积极开发中，功能可能会发生变化，使用前请充分测试。

---

## 📦 安装

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/xuyedev/saf.git
cd saf

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装 SAF
pip install -e .
```

### 验证安装

```bash
saf --help
```

---

## 🚀 快速开始

### 命令行工具

#### 1. 压缩文件

```bash
# 压缩 numpy 数组（.npy 文件）
saf compress data/sequences/fibonacci_10000.npy -o output.saf

# 输出：
# ✓ 压缩完成
# 算法：fibonacci
# 置信度：100.0%
# 压缩比：198.3×
# 原始大小：79.00 KB → 压缩后：404 B
```

#### 2. 解压文件

```bash
# 解压 .saf 文件
saf decompress output.saf -o restored.npy

# 输出：
# ✓ 解压完成
# 数据成功还原
```

#### 3. 验证无损性

```bash
# 验证数据完整性
saf verify data/sequences/fibonacci_10000.npy restored.npy

# 输出：
# ✓ 验证通过
# 哈希匹配：✓
# bit-level 无损：✓
```

#### 4. 批量性能测试

```bash
# 对目录进行批量测试
saf benchmark data/sequences/

# 输出：
# 正在处理 10 个文件...
# 平均压缩比：2195×
# 总耗时：33.09s
# 成功率：100%
```

### Python API

```python
from src.storage.compressor import Compressor
from src.storage.decompressor import Decompressor
import numpy as np

# 加载数据
data = np.load('data/sequences/fibonacci_10000.npy')

# 压缩
compressor = Compressor()
result = compressor.compress(data, 'output.saf')
print(f"压缩比：{result.compression_ratio:.1f}×")

# 解压
decompressor = Decompressor()
restored_data = decompressor.decompress('output.saf')

# 验证
assert np.array_equal(data, restored_data)
print("✓ bit-level 无损验证通过！")
```

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| **开发语言** | Python 3.10+ |
| **科学计算** | NumPy, SciPy |
| **符号计算** | SymPy |
| **CLI 框架** | Click |
| **序列化** | msgpack |
| **测试框架** | pytest, pytest-cov |
| **进度显示** | tqdm |
| **类型检查** | mypy |
| **代码格式化** | black, ruff |

---

## 📁 项目结构

```
saf/
├── src/                      # 源代码
│   ├── cli/                  # 命令行接口
│   ├── detectors/            # 模式检测器
│   ├── generators/           # 数据生成器
│   ├── storage/              # 压缩/解压引擎
│   ├── utils/                # 工具函数
│   └── verification/         # 验证与基准测试
├── tests/                    # 测试套件（169 个测试，87% 覆盖率）
├── data/                     # 示例数据
├── config/                   # 配置文件
├── docs/                     # 文档
├── LICENSE                   # MIT 许可证
└── README_CN.md              # 本文件
```

---

## 🧪 测试与质量保证

- **单元测试**：169 个测试（100% 通过）
- **测试覆盖率**：87%
- **集成测试**：端到端工作流验证
- **并发测试**：线程安全性验证（100% 成功）
- **大规模测试**：4K 图像（31.64 MB）、100 万项序列（7.63 MB）
- **类型检查**：mypy 严格模式
- **代码规范**：PEP 8 兼容（black + ruff）

---

## ⚠️ 已知限制

### 性能限制

- **Python GIL**：限制 CPU 密集型并发效率
- **小文件开销**：< 1KB 文件元数据开销较大
- **检测耗时**：大数据的模式检测可能需要数秒

### 不支持的数据类型

- π 数字序列（计算成本 > 存储成本）
- 真随机数据（白噪声）
- float64 分形（需要功能扩展）

### 推荐使用场景

- **最佳**：大规模批量处理（> 1MB 文件）
- **适合**：归档存储、带宽受限的数据传输
- **避免**：实时压缩、极小文件

---

## 📚 文档

- [用户指南](docs/user_guide.md) - 详细使用说明
- [API 参考](docs/api_reference.md) - Python API 文档
- [开发者指南](docs/developer_guide.md) - 架构与贡献指南
- [测试计划](测试进度/TEST_PLAN.md) - 全面测试结果
- [开发阶段](DEVELOPMENT_STAGES.md) - 项目路线图

---

## 💬 反馈与支持

本项目目前处于**测试和优化阶段**，您的反馈非常宝贵！

### 报告问题

- 🐛 **Bug 报告**：请在 GitHub 提交 issue 或通过邮箱联系
- 💡 **功能建议**：欢迎提出改进建议
- 📖 **文档改进**：帮助完善文档

### 联系方式

- **邮箱**：xu3033866090@gmail.com
- **微信**：xuyedev
- **GitHub Issues**：[https://github.com/xuyedev/saf/issues](https://github.com/xuyedev/saf/issues)

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

**重要提示**：本软件正在积极开发中，使用前请阅读 LICENSE 文件中的开发状态声明。

---

## 🙏 致谢

本项目基于 **Kolmogorov 复杂度** 理论，并受以下研究启发：

- **Kolmogorov 复杂度**：Li, M., & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications*. Springer.
- **符号回归**：Koza, J. R. (1992). *Genetic Programming*. MIT Press.
- **分形压缩**：Barnsley, M. F., & Hurd, L. P. (1993). *Fractal Image Compression*. AK Peters.

---

## 📊 项目统计

```
开发阶段：7/8 已完成
代码行数：~8,000
测试用例：169（100% 通过）
测试覆盖率：87%
支持算法：11 种（fibonacci, primes, polynomial, arithmetic, geometric,
              mandelbrot, julia, checkerboard, stripes, perlin_noise, gzip）
最高压缩比：82,123×
平均压缩比：2,195×
```

---

**作者**：徐野
**日期**：2025-11-23
**版本**：v0.2.0

---

⭐ **如果您觉得这个项目有用，请在 GitHub 上给它一个星标！**
