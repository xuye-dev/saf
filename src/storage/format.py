"""
算法存储格式定义

Author: 徐野
Date: 2025-11-23
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from ..utils.hash_utils import compute_hash
from ..utils.serialization import (
    deserialize_msgpack,
    load_msgpack,
    save_msgpack,
    serialize_msgpack,
)


class StorageMetadata(BaseModel):
    """存储元数据"""

    original_size: int = Field(..., description="原始数据大小（字节）", ge=0)
    original_hash: str = Field(..., description="原始数据 SHA-256 哈希值")
    data_type: str = Field(..., description="数据类型（sequence/image）")
    shape: Tuple[int, ...] = Field(..., description="数据形状")
    dtype: str = Field(..., description="数据类型（numpy dtype）")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """验证数据类型"""
        allowed_types = {"sequence", "image"}
        if v not in allowed_types:
            raise ValueError(f"数据类型必须是 {allowed_types} 之一，当前值: {v}")
        return v

    @field_validator("original_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """验证哈希值格式"""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"SHA-256 哈希值必须是 64 个十六进制字符，当前值: {v}")
        return v.lower()


class AlgorithmInfo(BaseModel):
    """算法信息"""

    type: str = Field(..., description="算法类型")
    parameters: Dict[str, Any] = Field(..., description="算法参数")
    code: Optional[str] = Field(default=None, description="生成代码（可选）")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """验证算法类型"""
        allowed_types = {
            # 序列算法
            "fibonacci",
            "recursive",  # 通用递归规律（包括斐波那契变种）
            "arithmetic",
            "geometric",
            "polynomial",
            "primes",
            "pi_digits",
            # 分形算法
            "mandelbrot",
            "julia",
            # 2D图案算法
            "checkerboard",
            "stripes",
            # 噪声算法
            "perlin_noise",
            # 降级压缩
            "gzip",
        }
        if v not in allowed_types:
            raise ValueError(f"算法类型必须是 {allowed_types} 之一，当前值: {v}")
        return v


class CompressionInfo(BaseModel):
    """压缩信息"""

    compressed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="压缩时间（ISO 8601 格式）",
    )
    confidence: float = Field(..., description="模式检测置信度（0-1）", ge=0, le=1)
    compression_time: float = Field(
        default=0.0, description="压缩耗时（秒）", ge=0
    )


class StorageFormat(BaseModel):
    """算法存储格式（完整结构）"""

    version: str = Field(default="1.0", description="格式版本号")
    metadata: StorageMetadata = Field(..., description="元数据")
    algorithm: AlgorithmInfo = Field(..., description="算法信息")
    compression_info: CompressionInfo = Field(..., description="压缩信息")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """验证版本号格式"""
        parts = v.split(".")
        if len(parts) < 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"版本号格式错误，应为 'X.Y' 或 'X.Y.Z' 格式，当前值: {v}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            "version": self.version,
            "metadata": {
                "original_size": self.metadata.original_size,
                "original_hash": self.metadata.original_hash,
                "data_type": self.metadata.data_type,
                "shape": list(self.metadata.shape),
                "dtype": self.metadata.dtype,
            },
            "algorithm": {
                "type": self.algorithm.type,
                "parameters": self.algorithm.parameters,
                "code": self.algorithm.code,
            },
            "compression_info": {
                "compressed_at": self.compression_info.compressed_at,
                "confidence": self.compression_info.confidence,
                "compression_time": self.compression_info.compression_time,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageFormat":
        """
        从字典创建 StorageFormat 实例

        Args:
            data: 字典数据

        Returns:
            StorageFormat: 存储格式实例
        """
        return cls(
            version=data.get("version", "1.0"),
            metadata=StorageMetadata(**data["metadata"]),
            algorithm=AlgorithmInfo(**data["algorithm"]),
            compression_info=CompressionInfo(**data["compression_info"]),
        )

    def serialize(self) -> bytes:
        """
        序列化为二进制数据（msgpack）

        Returns:
            bytes: 序列化后的字节数据
        """
        return serialize_msgpack(self.to_dict())

    @classmethod
    def deserialize(cls, data: bytes) -> "StorageFormat":
        """
        从二进制数据反序列化

        Args:
            data: 序列化后的字节数据

        Returns:
            StorageFormat: 存储格式实例
        """
        dict_data = deserialize_msgpack(data)
        return cls.from_dict(dict_data)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        保存到文件（.saf 格式）

        Args:
            file_path: 文件路径
        """
        save_msgpack(file_path, self.to_dict())

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "StorageFormat":
        """
        从文件加载（.saf 格式）

        Args:
            file_path: 文件路径

        Returns:
            StorageFormat: 存储格式实例
        """
        dict_data = load_msgpack(file_path)
        return cls.from_dict(dict_data)


def create_storage_format(
    original_data: bytes,
    data_type: str,
    shape: Tuple[int, ...],
    dtype: str,
    algorithm_type: str,
    algorithm_parameters: Dict[str, Any],
    confidence: float,
    compression_time: float = 0.0,
    algorithm_code: Optional[str] = None,
) -> StorageFormat:
    """
    创建存储格式实例（便捷函数）

    Args:
        original_data: 原始数据（字节）
        data_type: 数据类型（sequence/image）
        shape: 数据形状
        dtype: numpy dtype 字符串
        algorithm_type: 算法类型
        algorithm_parameters: 算法参数
        confidence: 检测置信度
        compression_time: 压缩耗时（秒）
        algorithm_code: 算法代码（可选）

    Returns:
        StorageFormat: 存储格式实例
    """
    # 计算原始数据哈希
    original_hash = compute_hash(original_data)

    # 创建元数据
    metadata = StorageMetadata(
        original_size=len(original_data),
        original_hash=original_hash,
        data_type=data_type,
        shape=shape,
        dtype=dtype,
    )

    # 创建算法信息
    algorithm = AlgorithmInfo(
        type=algorithm_type,
        parameters=algorithm_parameters,
        code=algorithm_code,
    )

    # 创建压缩信息
    compression_info = CompressionInfo(
        confidence=confidence,
        compression_time=compression_time,
    )

    return StorageFormat(
        metadata=metadata,
        algorithm=algorithm,
        compression_info=compression_info,
    )
