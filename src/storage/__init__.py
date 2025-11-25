"""
存储引擎模块（算法式存储）

Author: 徐野
Date: 2025-11-23
"""

from .compressor import AlgorithmicCompressor, compress_file
from .decompressor import (
    AlgorithmicDecompressor,
    decompress_file,
    decompress_to_array,
)
from .executor import AlgorithmExecutor, rebuild_data_from_file
from .format import (
    AlgorithmInfo,
    CompressionInfo,
    StorageFormat,
    StorageMetadata,
    create_storage_format,
)

__all__ = [
    # 压缩器
    "AlgorithmicCompressor",
    "compress_file",
    # 解压器
    "AlgorithmicDecompressor",
    "decompress_file",
    "decompress_to_array",
    # 执行器
    "AlgorithmExecutor",
    "rebuild_data_from_file",
    # 存储格式
    "StorageFormat",
    "StorageMetadata",
    "AlgorithmInfo",
    "CompressionInfo",
    "create_storage_format",
]
