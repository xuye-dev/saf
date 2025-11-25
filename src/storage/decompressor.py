"""
数据解压器（算法式存储）

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..models import VerificationResult
from ..utils.hash_utils import compute_hash, hash_comparison
from ..utils.logger import get_logger
from ..utils.timer import Timer
from .executor import AlgorithmExecutor
from .format import StorageFormat

logger = get_logger(__name__)


class AlgorithmicDecompressor:
    """
    算法式解压器

    核心流程：
    1. 读取 .saf 文件
    2. 解析存储格式
    3. 调用算法执行器重建数据
    4. 验证哈希值确保无损
    5. 返回重建的数据
    """

    def __init__(self, verify_hash: bool = True) -> None:
        """
        初始化解压器

        Args:
            verify_hash: 是否验证哈希值（默认 True）
        """
        self.verify_hash = verify_hash
        self.executor = AlgorithmExecutor()
        logger.info(f"算法式解压器已初始化，哈希验证: {verify_hash}")

    def decompress(
        self, input_path: Union[str, Path]
    ) -> Tuple[NDArray, VerificationResult]:
        """
        解压 .saf 文件并验证数据完整性

        Args:
            input_path: 输入文件路径（.saf 格式）

        Returns:
            Tuple[NDArray, VerificationResult]: (重建的数据, 验证结果)

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: 解压失败

        Example:
            >>> decompressor = AlgorithmicDecompressor()
            >>> data, verification = decompressor.decompress("data.saf")
            >>> print(verification.is_valid)
            True
        """
        input_path = Path(input_path)
        logger.info(f"开始解压文件: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")

        timer = Timer()
        timer.start()

        try:
            # 读取存储格式
            logger.debug("读取存储格式...")
            storage_format = StorageFormat.load(input_path)

            logger.info(
                f"存储格式: 版本={storage_format.version}, "
                f"算法类型={storage_format.algorithm.type}, "
                f"置信度={storage_format.compression_info.confidence:.2f}"
            )

            # 重建数据
            logger.info("开始重建数据...")
            data = self.executor.rebuild_data(storage_format)

            logger.info(f"数据重建完成，形状: {data.shape}, dtype: {data.dtype}")

            # 验证哈希值
            if self.verify_hash:
                logger.info("验证数据完整性...")
                verification = self._verify_data(
                    data=data,
                    expected_hash=storage_format.metadata.original_hash,
                    verification_time=timer.elapsed(),
                )

                if not verification.is_valid:
                    logger.error(
                        f"哈希验证失败: 期望={verification.original_hash[:16]}..., "
                        f"实际={verification.reconstructed_hash[:16]}..."
                    )
                    raise RuntimeError(
                        f"数据完整性验证失败: {verification.error_message}"
                    )

                logger.info("哈希验证通过，数据完整")
            else:
                # 跳过验证，但仍计算哈希值以满足模型要求
                logger.warning("跳过哈希验证")
                reconstructed_hash = compute_hash(data.tobytes())
                verification = VerificationResult(
                    is_valid=True,
                    original_hash=storage_format.metadata.original_hash,
                    reconstructed_hash=reconstructed_hash,
                    verification_time=timer.elapsed(),
                )

            return data, verification

        except Exception as e:
            logger.error(f"解压失败: {e}")
            raise RuntimeError(f"解压失败: {e}") from e

    def _verify_data(
        self, data: NDArray, expected_hash: str, verification_time: float
    ) -> VerificationResult:
        """
        验证数据完整性

        Args:
            data: 重建的数据
            expected_hash: 期望的哈希值
            verification_time: 验证耗时

        Returns:
            VerificationResult: 验证结果
        """
        # 计算重建数据的哈希
        reconstructed_hash = compute_hash(data.tobytes())

        # 比较哈希值
        is_valid = hash_comparison(reconstructed_hash, expected_hash)

        if is_valid:
            return VerificationResult(
                is_valid=True,
                original_hash=expected_hash,
                reconstructed_hash=reconstructed_hash,
                verification_time=verification_time,
            )
        else:
            return VerificationResult(
                is_valid=False,
                original_hash=expected_hash,
                reconstructed_hash=reconstructed_hash,
                error_message=f"哈希值不匹配: 期望={expected_hash}, 实际={reconstructed_hash}",
                verification_time=verification_time,
            )


def decompress_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    verify_hash: bool = True,
) -> VerificationResult:
    """
    解压文件并保存为 .npy 格式

    Args:
        input_path: 输入文件路径（.saf 格式）
        output_path: 输出文件路径（.npy 格式）
        verify_hash: 是否验证哈希值

    Returns:
        VerificationResult: 验证结果

    Example:
        >>> verification = decompress_file("data.saf", "data.npy")
        >>> print(verification.is_valid)
        True
    """
    output_path = Path(output_path)

    # 解压数据
    decompressor = AlgorithmicDecompressor(verify_hash=verify_hash)
    data, verification = decompressor.decompress(input_path)

    # 保存数据
    logger.info(f"保存数据到: {output_path}")
    np.save(output_path, data)

    return verification


def decompress_to_array(
    input_path: Union[str, Path], verify_hash: bool = True
) -> NDArray:
    """
    解压文件并直接返回 NumPy 数组（便捷函数）

    Args:
        input_path: 输入文件路径（.saf 格式）
        verify_hash: 是否验证哈希值

    Returns:
        NDArray: 重建的数据

    Raises:
        RuntimeError: 哈希验证失败

    Example:
        >>> data = decompress_to_array("data.saf")
        >>> print(data.shape)
    """
    decompressor = AlgorithmicDecompressor(verify_hash=verify_hash)
    data, verification = decompressor.decompress(input_path)

    if not verification.is_valid:
        raise RuntimeError(f"哈希验证失败: {verification.error_message}")

    return data
