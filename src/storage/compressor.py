"""
数据压缩器（算法式存储）

Author: 徐野
Date: 2025-11-23
"""

import gzip
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from ..detectors.pattern_matcher import PatternMatcher
from ..models import CompressionResult
from ..utils.hash_utils import compute_hash
from ..utils.logger import get_logger
from ..utils.timer import Timer
from .format import StorageFormat, create_storage_format

logger = get_logger(__name__)


class AlgorithmicCompressor:
    """
    算法式压缩器

    核心流程：
    1. 检测数据的生成规律（使用 PatternMatcher）
    2. 如果检测成功且置信度高，提取算法参数进行算法式存储
    3. 如果检测失败或置信度低，回退到 gzip 压缩
    4. 生成 .saf 文件并返回压缩结果
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        """
        初始化压缩器

        Args:
            confidence_threshold: 置信度阈值（低于此值将回退到 gzip）
        """
        self.confidence_threshold = confidence_threshold
        self.pattern_matcher = PatternMatcher(confidence_threshold=confidence_threshold)
        logger.info(f"算法式压缩器已初始化，置信度阈值: {confidence_threshold}")

    def compress(
        self,
        data: NDArray[Any],
        output_path: Union[str, Path],
        force_gzip: bool = False,
    ) -> CompressionResult:
        """
        压缩数据并保存为 .saf 文件

        Args:
            data: 待压缩的数据（NumPy 数组）
            output_path: 输出文件路径（.saf 格式）
            force_gzip: 是否强制使用 gzip 压缩（跳过模式检测）

        Returns:
            CompressionResult: 压缩结果

        Example:
            >>> compressor = AlgorithmicCompressor()
            >>> data = np.array([0, 1, 1, 2, 3, 5, 8, 13])
            >>> result = compressor.compress(data, "output.saf")
            >>> print(result.compression_ratio)
        """
        timer = Timer()
        timer.start()

        output_path = Path(output_path)
        logger.info(f"开始压缩数据，形状: {data.shape}, 输出: {output_path}")

        # 计算原始数据大小和哈希
        original_data_bytes = data.tobytes()
        original_size = len(original_data_bytes)
        original_hash = compute_hash(original_data_bytes)

        logger.debug(f"原始数据大小: {original_size} 字节, 哈希: {original_hash[:16]}...")

        # 尝试算法式压缩（除非强制 gzip）
        if not force_gzip:
            try:
                logger.info("尝试模式检测...")
                pattern = self.pattern_matcher.detect(data)

                logger.info(
                    f"模式检测完成: 类型={pattern.pattern_type}, 置信度={pattern.confidence:.2f}"
                )

                # 如果置信度足够高，使用算法式存储
                if pattern.confidence >= self.confidence_threshold:
                    storage_format = self._create_algorithmic_storage(
                        data=data,
                        original_data_bytes=original_data_bytes,
                        pattern=pattern,
                        compression_time=timer.elapsed(),
                    )

                    # 保存到文件
                    storage_format.save(output_path)
                    compressed_size = output_path.stat().st_size

                    logger.info(
                        f"算法式压缩成功，压缩后大小: {compressed_size} 字节"
                    )

                    return CompressionResult(
                        original_size=original_size,
                        compressed_size=compressed_size,
                        compression_ratio=original_size / compressed_size,
                        algorithm_type=pattern.pattern_type,
                        original_hash=original_hash,
                        compression_time=timer.elapsed(),
                        metadata={
                            "confidence": pattern.confidence,
                            "method": "algorithmic",
                        },
                    )
                else:
                    logger.warning(
                        f"置信度过低 ({pattern.confidence:.2f} < "
                        f"{self.confidence_threshold})，回退到 gzip"
                    )

            except Exception as e:
                logger.warning(f"模式检测失败: {e}，回退到 gzip")

        # 降级策略：使用 gzip 压缩
        logger.info("使用 gzip 压缩...")
        storage_format = self._create_gzip_storage(
            data=data,
            original_data_bytes=original_data_bytes,
            compression_time=timer.elapsed(),
        )

        # 保存到文件
        storage_format.save(output_path)
        compressed_size = output_path.stat().st_size

        logger.info(f"gzip 压缩完成，压缩后大小: {compressed_size} 字节")

        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            algorithm_type="gzip",
            original_hash=original_hash,
            compression_time=timer.elapsed(),
            metadata={"method": "gzip_fallback"},
        )

    def _create_algorithmic_storage(
        self,
        data: NDArray[Any],
        original_data_bytes: bytes,
        pattern: Any,
        compression_time: float,
    ) -> StorageFormat:
        """创建算法式存储格式"""
        # 确定数据类型
        data_type = "image" if data.ndim == 2 else "sequence"

        # 提取算法类型和参数
        algorithm_type = pattern.pattern_type
        algorithm_parameters = pattern.parameters.copy()

        # 添加序列长度参数（用于重建数据）
        if data_type == "sequence":
            algorithm_parameters["n"] = len(data)

        logger.debug(f"算法类型: {algorithm_type}, 参数: {algorithm_parameters}")

        return create_storage_format(
            original_data=original_data_bytes,
            data_type=data_type,
            shape=data.shape,
            dtype=str(data.dtype),
            algorithm_type=algorithm_type,
            algorithm_parameters=algorithm_parameters,
            confidence=pattern.confidence,
            compression_time=compression_time,
        )

    def _create_gzip_storage(
        self,
        data: NDArray[Any],
        original_data_bytes: bytes,
        compression_time: float,
    ) -> StorageFormat:
        """创建 gzip 压缩存储格式"""
        # 压缩数据
        compressed_data = gzip.compress(original_data_bytes, compresslevel=9)

        # 确定数据类型
        data_type = "image" if data.ndim == 2 else "sequence"

        # 构建 gzip 参数（包含压缩后的数据）
        algorithm_parameters = {
            "compressed_data": list(compressed_data),  # 转换为列表以便 msgpack 序列化
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        }

        logger.debug(f"gzip 压缩数据大小: {len(compressed_data)} 字节")

        return create_storage_format(
            original_data=original_data_bytes,
            data_type=data_type,
            shape=data.shape,
            dtype=str(data.dtype),
            algorithm_type="gzip",
            algorithm_parameters=algorithm_parameters,
            confidence=1.0,  # gzip 是无损压缩，置信度为 1.0
            compression_time=compression_time,
        )


def compress_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    confidence_threshold: float = 0.85,
) -> CompressionResult:
    """
    压缩文件（从 .npy 文件读取数据并压缩为 .saf 文件）

    Args:
        input_path: 输入文件路径（.npy 格式）
        output_path: 输出文件路径（.saf 格式）
        confidence_threshold: 置信度阈值

    Returns:
        CompressionResult: 压缩结果

    Example:
        >>> result = compress_file("data.npy", "data.saf")
        >>> print(f"压缩比: {result.compression_ratio:.2f}x")
    """
    input_path = Path(input_path)

    # 读取数据
    logger.info(f"读取数据文件: {input_path}")
    data = np.load(input_path)

    # 压缩数据
    compressor = AlgorithmicCompressor(confidence_threshold=confidence_threshold)
    return compressor.compress(data, output_path)
