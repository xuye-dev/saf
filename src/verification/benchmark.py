"""
性能基准测试模块

提供压缩/解压性能测试功能，包括：
- 压缩时间统计
- 解压时间统计
- 压缩比统计
- 内存使用监控

Author: 徐野
Date: 2025-11-23
"""

import gc
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..storage.compressor import AlgorithmicCompressor
from ..storage.decompressor import AlgorithmicDecompressor
from ..utils.logger import get_logger
from ..utils.timer import Timer, format_time

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """
    单个基准测试结果

    Attributes:
        name: 测试名称
        original_size: 原始数据大小（字节）
        compressed_size: 压缩后大小（字节）
        compression_ratio: 压缩比
        compression_time: 压缩耗时（秒）
        decompression_time: 解压耗时（秒）
        peak_memory: 峰值内存使用（字节）
        algorithm_type: 算法类型
        is_valid: 验证是否通过
        timestamp: 测试时间戳
        metadata: 额外元数据
    """

    name: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    peak_memory: int
    algorithm_type: str
    is_valid: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "compression_time": self.compression_time,
            "decompression_time": self.decompression_time,
            "peak_memory": self.peak_memory,
            "algorithm_type": self.algorithm_type,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"BenchmarkResult({self.name}):\n"
            f"  原始大小: {self.original_size:,} bytes\n"
            f"  压缩大小: {self.compressed_size:,} bytes\n"
            f"  压缩比: {self.compression_ratio:.2f}x\n"
            f"  压缩耗时: {format_time(self.compression_time)}\n"
            f"  解压耗时: {format_time(self.decompression_time)}\n"
            f"  峰值内存: {self.peak_memory / 1024 / 1024:.2f} MB\n"
            f"  算法类型: {self.algorithm_type}\n"
            f"  验证通过: {self.is_valid}"
        )


@dataclass
class BenchmarkSummary:
    """
    基准测试汇总结果

    Attributes:
        total_tests: 总测试数
        passed_tests: 通过测试数
        failed_tests: 失败测试数
        avg_compression_ratio: 平均压缩比
        avg_compression_time: 平均压缩时间
        avg_decompression_time: 平均解压时间
        total_original_size: 原始数据总大小
        total_compressed_size: 压缩后总大小
        results: 详细测试结果列表
    """

    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_compression_ratio: float
    avg_compression_time: float
    avg_decompression_time: float
    total_original_size: int
    total_compressed_size: int
    results: List[BenchmarkResult]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_compression_time": self.avg_compression_time,
            "avg_decompression_time": self.avg_decompression_time,
            "total_original_size": self.total_original_size,
            "total_compressed_size": self.total_compressed_size,
            "results": [r.to_dict() for r in self.results],
        }

    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"=== 基准测试汇总 ===\n"
            f"总测试数: {self.total_tests}\n"
            f"通过: {self.passed_tests}, 失败: {self.failed_tests}\n"
            f"平均压缩比: {self.avg_compression_ratio:.2f}x\n"
            f"平均压缩时间: {format_time(self.avg_compression_time)}\n"
            f"平均解压时间: {format_time(self.avg_decompression_time)}\n"
            f"总原始大小: {self.total_original_size / 1024 / 1024:.2f} MB\n"
            f"总压缩大小: {self.total_compressed_size / 1024 / 1024:.2f} MB\n"
            f"总体压缩比: {self.total_original_size / max(self.total_compressed_size, 1):.2f}x"
        )


class PerformanceBenchmark:
    """
    性能基准测试器

    支持：
    - 单文件基准测试
    - 批量文件基准测试
    - 内存使用监控
    - 详细性能报告

    Example:
        >>> benchmark = PerformanceBenchmark()
        >>> result = benchmark.run_single("data.npy", "output.saf")
        >>> print(result)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        初始化基准测试器

        Args:
            confidence_threshold: 置信度阈值
            output_dir: 输出目录（用于临时文件）
        """
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compressor = AlgorithmicCompressor(confidence_threshold=confidence_threshold)
        self.decompressor = AlgorithmicDecompressor(verify_hash=True)

        logger.info(f"性能基准测试器已初始化，输出目录: {self.output_dir}")

    def run_single(
        self,
        data: Union[NDArray[Any], str, Path],
        name: Optional[str] = None,
        cleanup: bool = True,
    ) -> BenchmarkResult:
        """
        运行单个基准测试

        Args:
            data: 测试数据（NumPy 数组或 .npy 文件路径）
            name: 测试名称（可选）
            cleanup: 是否清理临时文件

        Returns:
            BenchmarkResult: 基准测试结果
        """
        # 加载数据
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            name = name or data_path.stem
            data_array = np.load(data_path)
        else:
            data_array = data
            name = name or f"data_{data_array.shape}"

        logger.info(f"开始基准测试: {name}, 形状: {data_array.shape}")

        # 准备输出路径
        saf_path = self.output_dir / f"{name}_benchmark.saf"

        # 内存监控开始
        gc.collect()
        tracemalloc.start()

        try:
            # 压缩测试
            compression_timer = Timer()
            compression_timer.start()
            compression_result = self.compressor.compress(data_array, saf_path)
            compression_time = compression_timer.stop()

            # 解压测试
            decompression_timer = Timer()
            decompression_timer.start()
            reconstructed_data, verification = self.decompressor.decompress(saf_path)
            decompression_time = decompression_timer.stop()

            # 获取内存峰值
            _, peak_memory = tracemalloc.get_traced_memory()

            # 验证数据
            is_valid = verification.is_valid and np.array_equal(data_array, reconstructed_data)

            result = BenchmarkResult(
                name=name,
                original_size=compression_result.original_size,
                compressed_size=compression_result.compressed_size,
                compression_ratio=compression_result.compression_ratio,
                compression_time=compression_time,
                decompression_time=decompression_time,
                peak_memory=peak_memory,
                algorithm_type=compression_result.algorithm_type,
                is_valid=is_valid,
                metadata={
                    "shape": list(data_array.shape),
                    "dtype": str(data_array.dtype),
                    "confidence": compression_result.metadata.get("confidence", 1.0),
                },
            )

            logger.info(f"基准测试完成: {name}, 压缩比: {result.compression_ratio:.2f}x")

            return result

        finally:
            tracemalloc.stop()

            # 清理临时文件
            if cleanup and saf_path.exists():
                saf_path.unlink()
                logger.debug(f"已清理临时文件: {saf_path}")

    def run_batch(
        self,
        data_sources: List[Union[NDArray[Any], str, Path]],
        names: Optional[List[str]] = None,
        cleanup: bool = True,
    ) -> BenchmarkSummary:
        """
        批量运行基准测试

        Args:
            data_sources: 测试数据列表
            names: 测试名称列表（可选）
            cleanup: 是否清理临时文件

        Returns:
            BenchmarkSummary: 基准测试汇总结果
        """
        results: List[BenchmarkResult] = []
        total = len(data_sources)

        logger.info(f"开始批量基准测试，共 {total} 个数据源")

        for i, data in enumerate(data_sources):
            name = names[i] if names and i < len(names) else None

            try:
                result = self.run_single(data, name=name, cleanup=cleanup)
                results.append(result)
            except Exception as e:
                logger.error(f"基准测试失败 [{i + 1}/{total}]: {e}")
                # 创建失败结果
                results.append(
                    BenchmarkResult(
                        name=name or f"test_{i}",
                        original_size=0,
                        compressed_size=0,
                        compression_ratio=0.0,
                        compression_time=0.0,
                        decompression_time=0.0,
                        peak_memory=0,
                        algorithm_type="error",
                        is_valid=False,
                        metadata={"error": str(e)},
                    )
                )

        return self._create_summary(results)

    def run_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.npy",
        cleanup: bool = True,
    ) -> BenchmarkSummary:
        """
        对目录中的所有数据文件运行基准测试

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            cleanup: 是否清理临时文件

        Returns:
            BenchmarkSummary: 基准测试汇总结果
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        data_files = sorted(directory.glob(pattern))
        logger.info(f"在 {directory} 中找到 {len(data_files)} 个数据文件")

        return self.run_batch(
            data_sources=[str(f) for f in data_files],
            names=[f.stem for f in data_files],
            cleanup=cleanup,
        )

    def _create_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """
        创建基准测试汇总

        Args:
            results: 测试结果列表

        Returns:
            BenchmarkSummary: 汇总结果
        """
        if not results:
            return BenchmarkSummary(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                avg_compression_ratio=0.0,
                avg_compression_time=0.0,
                avg_decompression_time=0.0,
                total_original_size=0,
                total_compressed_size=0,
                results=[],
            )

        valid_results = [r for r in results if r.is_valid and r.original_size > 0]

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.is_valid)
        failed_tests = total_tests - passed_tests

        total_original_size = sum(r.original_size for r in results)
        total_compressed_size = sum(r.compressed_size for r in results)

        # 计算平均值（仅基于有效结果）
        if valid_results:
            avg_compression_ratio = sum(r.compression_ratio for r in valid_results) / len(
                valid_results
            )
            avg_compression_time = sum(r.compression_time for r in valid_results) / len(
                valid_results
            )
            avg_decompression_time = sum(r.decompression_time for r in valid_results) / len(
                valid_results
            )
        else:
            avg_compression_ratio = 0.0
            avg_compression_time = 0.0
            avg_decompression_time = 0.0

        return BenchmarkSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            avg_compression_ratio=avg_compression_ratio,
            avg_compression_time=avg_compression_time,
            avg_decompression_time=avg_decompression_time,
            total_original_size=total_original_size,
            total_compressed_size=total_compressed_size,
            results=results,
        )


class MemoryProfiler:
    """
    内存分析器

    用于监控压缩/解压过程中的内存使用

    Example:
        >>> profiler = MemoryProfiler()
        >>> with profiler:
        ...     # 执行操作
        ...     pass
        >>> print(profiler.peak_memory)
    """

    def __init__(self) -> None:
        """初始化内存分析器"""
        self.peak_memory: int = 0
        self.current_memory: int = 0
        self._is_tracing: bool = False

    def start(self) -> "MemoryProfiler":
        """开始内存监控"""
        gc.collect()
        tracemalloc.start()
        self._is_tracing = True
        return self

    def stop(self) -> int:
        """
        停止内存监控

        Returns:
            int: 峰值内存使用（字节）
        """
        if self._is_tracing:
            self.current_memory, self.peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._is_tracing = False
        return self.peak_memory

    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息

        Returns:
            Dict[str, Any]: 内存统计
        """
        return {
            "current_memory_bytes": self.current_memory,
            "peak_memory_bytes": self.peak_memory,
            "current_memory_mb": self.current_memory / 1024 / 1024,
            "peak_memory_mb": self.peak_memory / 1024 / 1024,
        }

    def __enter__(self) -> "MemoryProfiler":
        """上下文管理器入口"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """上下文管理器出口"""
        self.stop()


def benchmark_file(
    data_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> BenchmarkResult:
    """
    对单个文件进行基准测试（便捷函数）

    Args:
        data_path: 数据文件路径（.npy 格式）
        output_dir: 输出目录

    Returns:
        BenchmarkResult: 基准测试结果

    Example:
        >>> result = benchmark_file("data/sequences/fibonacci_10000.npy")
        >>> print(f"压缩比: {result.compression_ratio:.2f}x")
    """
    benchmark = PerformanceBenchmark(output_dir=output_dir)
    return benchmark.run_single(data_path)


def benchmark_directory(
    directory: Union[str, Path],
    pattern: str = "*.npy",
    output_dir: Optional[Union[str, Path]] = None,
) -> BenchmarkSummary:
    """
    对目录中的所有文件进行基准测试（便捷函数）

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        output_dir: 输出目录

    Returns:
        BenchmarkSummary: 基准测试汇总

    Example:
        >>> summary = benchmark_directory("data/sequences/")
        >>> print(summary)
    """
    benchmark = PerformanceBenchmark(output_dir=output_dir)
    return benchmark.run_directory(directory, pattern=pattern)
