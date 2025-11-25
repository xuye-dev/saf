"""
无损性验证器模块

提供数据完整性验证功能，包括：
- 基于 SHA-256 的哈希验证
- 逐字节对比验证
- 验证报告生成

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..models import VerificationResult
from ..storage.decompressor import AlgorithmicDecompressor
from ..storage.format import StorageFormat
from ..utils.hash_utils import compute_hash
from ..utils.logger import get_logger
from ..utils.timer import Timer

logger = get_logger(__name__)


class DataVerifier:
    """
    数据无损性验证器

    支持多种验证方式：
    1. 哈希验证（SHA-256）
    2. 逐字节对比验证
    3. 完整的压缩-解压往返验证

    Example:
        >>> verifier = DataVerifier()
        >>> result = verifier.verify_file("data.saf", "original.npy")
        >>> print(result.is_valid)
        True
    """

    def __init__(self) -> None:
        """初始化验证器"""
        self.decompressor = AlgorithmicDecompressor(verify_hash=False)
        logger.info("数据验证器已初始化")

    def verify_hash(
        self,
        original_data: Union[bytes, NDArray[Any]],
        reconstructed_data: Union[bytes, NDArray[Any]],
    ) -> VerificationResult:
        """
        基于 SHA-256 哈希的数据验证

        Args:
            original_data: 原始数据（字节或 NumPy 数组）
            reconstructed_data: 重建数据（字节或 NumPy 数组）

        Returns:
            VerificationResult: 验证结果
        """
        timer = Timer()
        timer.start()

        # 转换为字节
        if isinstance(original_data, np.ndarray):
            original_bytes = original_data.tobytes()
        else:
            original_bytes = original_data

        if isinstance(reconstructed_data, np.ndarray):
            reconstructed_bytes = reconstructed_data.tobytes()
        else:
            reconstructed_bytes = reconstructed_data

        # 计算哈希
        original_hash = compute_hash(original_bytes)
        reconstructed_hash = compute_hash(reconstructed_bytes)

        # 比较哈希
        is_valid = original_hash == reconstructed_hash
        elapsed = timer.stop()

        if is_valid:
            logger.info(f"哈希验证通过，耗时: {elapsed:.4f}s")
        else:
            logger.error(
                f"哈希验证失败: 原始={original_hash[:16]}..., "
                f"重建={reconstructed_hash[:16]}..."
            )

        return VerificationResult(
            is_valid=is_valid,
            original_hash=original_hash,
            reconstructed_hash=reconstructed_hash,
            error_message=None if is_valid else "哈希值不匹配",
            verification_time=elapsed,
        )

    def verify_byte_by_byte(
        self,
        original_data: Union[bytes, NDArray[Any]],
        reconstructed_data: Union[bytes, NDArray[Any]],
    ) -> VerificationResult:
        """
        逐字节对比验证

        提供更详细的错误信息，包括第一个不匹配的位置

        Args:
            original_data: 原始数据（字节或 NumPy 数组）
            reconstructed_data: 重建数据（字节或 NumPy 数组）

        Returns:
            VerificationResult: 验证结果（包含差异位置信息）
        """
        timer = Timer()
        timer.start()

        # 转换为字节
        if isinstance(original_data, np.ndarray):
            original_bytes = original_data.tobytes()
        else:
            original_bytes = original_data

        if isinstance(reconstructed_data, np.ndarray):
            reconstructed_bytes = reconstructed_data.tobytes()
        else:
            reconstructed_bytes = reconstructed_data

        # 计算哈希（用于报告）
        original_hash = compute_hash(original_bytes)
        reconstructed_hash = compute_hash(reconstructed_bytes)

        # 检查长度
        if len(original_bytes) != len(reconstructed_bytes):
            elapsed = timer.stop()
            error_msg = (
                f"数据长度不匹配: 原始={len(original_bytes)}, "
                f"重建={len(reconstructed_bytes)}"
            )
            logger.error(error_msg)
            return VerificationResult(
                is_valid=False,
                original_hash=original_hash,
                reconstructed_hash=reconstructed_hash,
                error_message=error_msg,
                verification_time=elapsed,
            )

        # 逐字节对比
        for i, (o, r) in enumerate(zip(original_bytes, reconstructed_bytes)):
            if o != r:
                elapsed = timer.stop()
                error_msg = (
                    f"字节不匹配: 位置={i}, 原始=0x{o:02x}, 重建=0x{r:02x}"
                )
                logger.error(error_msg)
                return VerificationResult(
                    is_valid=False,
                    original_hash=original_hash,
                    reconstructed_hash=reconstructed_hash,
                    error_message=error_msg,
                    verification_time=elapsed,
                )

        elapsed = timer.stop()
        logger.info(f"逐字节验证通过，共验证 {len(original_bytes)} 字节，耗时: {elapsed:.4f}s")

        return VerificationResult(
            is_valid=True,
            original_hash=original_hash,
            reconstructed_hash=reconstructed_hash,
            verification_time=elapsed,
        )

    def verify_saf_file(
        self,
        saf_path: Union[str, Path],
        original_data: Optional[Union[bytes, NDArray[Any]]] = None,
        original_path: Optional[Union[str, Path]] = None,
        method: str = "hash",
    ) -> VerificationResult:
        """
        验证 .saf 压缩文件的完整性

        Args:
            saf_path: .saf 文件路径
            original_data: 原始数据（可选，与 original_path 二选一）
            original_path: 原始数据文件路径（.npy 格式，可选）
            method: 验证方法 ("hash" 或 "byte_by_byte")

        Returns:
            VerificationResult: 验证结果

        Raises:
            ValueError: 未提供原始数据或路径
            FileNotFoundError: 文件不存在

        Example:
            >>> verifier = DataVerifier()
            >>> result = verifier.verify_saf_file("data.saf", original_path="data.npy")
            >>> print(result.is_valid)
        """
        saf_path = Path(saf_path)
        logger.info(f"开始验证 .saf 文件: {saf_path}")

        if not saf_path.exists():
            raise FileNotFoundError(f"文件不存在: {saf_path}")

        # 获取原始数据
        if original_data is None and original_path is None:
            # 从 .saf 文件中读取存储的哈希值进行验证
            logger.info("未提供原始数据，使用存储的哈希值进行验证")
            return self._verify_with_stored_hash(saf_path)

        if original_path is not None:
            original_path = Path(original_path)
            if not original_path.exists():
                raise FileNotFoundError(f"原始数据文件不存在: {original_path}")
            original_data = np.load(original_path)

        # 解压数据
        reconstructed_data, _ = self.decompressor.decompress(saf_path)

        # 检查解压结果
        if reconstructed_data is None:
            return VerificationResult(
                is_valid=False,
                original_hash="",
                reconstructed_hash="",
                error_message="解压失败：无法重建数据",
                verification_time=0.0,
            )

        # 检查原始数据（逻辑上不应为 None，但显式检查以满足类型系统）
        if original_data is None:
            return VerificationResult(
                is_valid=False,
                original_hash="",
                reconstructed_hash="",
                error_message="原始数据为空",
                verification_time=0.0,
            )

        # 选择验证方法
        if method == "byte_by_byte":
            return self.verify_byte_by_byte(original_data, reconstructed_data)
        else:
            return self.verify_hash(original_data, reconstructed_data)

    def _verify_with_stored_hash(
        self, saf_path: Union[str, Path]
    ) -> VerificationResult:
        """
        使用 .saf 文件中存储的哈希值进行验证

        Args:
            saf_path: .saf 文件路径

        Returns:
            VerificationResult: 验证结果
        """
        timer = Timer()
        timer.start()

        saf_path = Path(saf_path)

        # 读取存储格式
        storage_format = StorageFormat.load(saf_path)
        stored_hash = storage_format.metadata.original_hash

        # 解压数据
        reconstructed_data, _ = self.decompressor.decompress(saf_path)
        reconstructed_hash = compute_hash(reconstructed_data.tobytes())

        # 比较哈希
        is_valid = stored_hash == reconstructed_hash
        elapsed = timer.stop()

        if is_valid:
            logger.info(f"哈希验证通过（使用存储哈希），耗时: {elapsed:.4f}s")
        else:
            logger.error(
                f"哈希验证失败: 存储={stored_hash[:16]}..., "
                f"重建={reconstructed_hash[:16]}..."
            )

        return VerificationResult(
            is_valid=is_valid,
            original_hash=stored_hash,
            reconstructed_hash=reconstructed_hash,
            error_message=None if is_valid else "哈希值与存储值不匹配",
            verification_time=elapsed,
        )

    def verify_roundtrip(
        self,
        original_data: NDArray[Any],
        saf_path: Union[str, Path],
        method: str = "hash",
    ) -> VerificationResult:
        """
        完整的压缩-解压往返验证

        验证数据经过压缩和解压后是否保持完全一致

        Args:
            original_data: 原始数据（NumPy 数组）
            saf_path: .saf 文件路径
            method: 验证方法 ("hash" 或 "byte_by_byte")

        Returns:
            VerificationResult: 验证结果
        """
        logger.info("执行压缩-解压往返验证")

        # 解压数据
        reconstructed_data, _ = self.decompressor.decompress(saf_path)

        # 验证形状
        if original_data.shape != reconstructed_data.shape:
            return VerificationResult(
                is_valid=False,
                original_hash=compute_hash(original_data.tobytes()),
                reconstructed_hash=compute_hash(reconstructed_data.tobytes()),
                error_message=(
                    f"形状不匹配: 原始={original_data.shape}, "
                    f"重建={reconstructed_data.shape}"
                ),
                verification_time=0.0,
            )

        # 验证数据类型
        if original_data.dtype != reconstructed_data.dtype:
            return VerificationResult(
                is_valid=False,
                original_hash=compute_hash(original_data.tobytes()),
                reconstructed_hash=compute_hash(reconstructed_data.tobytes()),
                error_message=(
                    f"数据类型不匹配: 原始={original_data.dtype}, "
                    f"重建={reconstructed_data.dtype}"
                ),
                verification_time=0.0,
            )

        # 选择验证方法
        if method == "byte_by_byte":
            return self.verify_byte_by_byte(original_data, reconstructed_data)
        else:
            return self.verify_hash(original_data, reconstructed_data)


class BatchVerifier:
    """
    批量验证器

    支持批量验证多个 .saf 文件

    Example:
        >>> verifier = BatchVerifier()
        >>> results = verifier.verify_directory("data/compressed/")
        >>> for path, result in results.items():
        ...     print(f"{path}: {'通过' if result.is_valid else '失败'}")
    """

    def __init__(self) -> None:
        """初始化批量验证器"""
        self.verifier = DataVerifier()
        logger.info("批量验证器已初始化")

    def verify_files(
        self,
        saf_files: List[Union[str, Path]],
        original_files: Optional[List[Union[str, Path]]] = None,
        method: str = "hash",
    ) -> Dict[str, VerificationResult]:
        """
        批量验证多个 .saf 文件

        Args:
            saf_files: .saf 文件路径列表
            original_files: 原始数据文件路径列表（可选，与 saf_files 一一对应）
            method: 验证方法 ("hash" 或 "byte_by_byte")

        Returns:
            Dict[str, VerificationResult]: 文件路径到验证结果的映射
        """
        results: Dict[str, VerificationResult] = {}
        total = len(saf_files)

        logger.info(f"开始批量验证 {total} 个文件")

        for i, saf_path in enumerate(saf_files):
            saf_path = Path(saf_path)
            logger.info(f"验证进度: {i + 1}/{total} - {saf_path.name}")

            try:
                original_path = None
                if original_files is not None and i < len(original_files):
                    original_path = original_files[i]

                result = self.verifier.verify_saf_file(
                    saf_path=saf_path,
                    original_path=original_path,
                    method=method,
                )
                results[str(saf_path)] = result

            except Exception as e:
                logger.error(f"验证失败 {saf_path}: {e}")
                results[str(saf_path)] = VerificationResult(
                    is_valid=False,
                    original_hash="error",
                    reconstructed_hash="error",
                    error_message=str(e),
                    verification_time=0.0,
                )

        # 统计结果
        passed = sum(1 for r in results.values() if r.is_valid)
        failed = total - passed
        logger.info(f"批量验证完成: 通过={passed}, 失败={failed}, 总计={total}")

        return results

    def verify_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.saf",
        method: str = "hash",
    ) -> Dict[str, VerificationResult]:
        """
        验证目录中的所有 .saf 文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式（默认 "*.saf"）
            method: 验证方法 ("hash" 或 "byte_by_byte")

        Returns:
            Dict[str, VerificationResult]: 文件路径到验证结果的映射
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        saf_files: List[Union[str, Path]] = list(directory.glob(pattern))
        logger.info(f"在 {directory} 中找到 {len(saf_files)} 个文件")

        return self.verify_files(saf_files, method=method)


def verify_data(
    original: Union[bytes, NDArray[Any]],
    reconstructed: Union[bytes, NDArray[Any]],
    method: str = "hash",
) -> VerificationResult:
    """
    验证数据一致性（便捷函数）

    Args:
        original: 原始数据
        reconstructed: 重建数据
        method: 验证方法 ("hash" 或 "byte_by_byte")

    Returns:
        VerificationResult: 验证结果

    Example:
        >>> result = verify_data(original_array, reconstructed_array)
        >>> print(result.is_valid)
    """
    verifier = DataVerifier()
    if method == "byte_by_byte":
        return verifier.verify_byte_by_byte(original, reconstructed)
    return verifier.verify_hash(original, reconstructed)


def verify_file(
    saf_path: Union[str, Path],
    original_path: Optional[Union[str, Path]] = None,
    method: str = "hash",
) -> VerificationResult:
    """
    验证 .saf 文件（便捷函数）

    Args:
        saf_path: .saf 文件路径
        original_path: 原始数据文件路径（可选）
        method: 验证方法 ("hash" 或 "byte_by_byte")

    Returns:
        VerificationResult: 验证结果

    Example:
        >>> result = verify_file("data.saf", "data.npy")
        >>> print(result.is_valid)
    """
    verifier = DataVerifier()
    return verifier.verify_saf_file(saf_path, original_path=original_path, method=method)
