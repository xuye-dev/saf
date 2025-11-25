"""
核心数据模型定义（使用 Pydantic）

Author: 徐野
Date: 2025-11-23
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class CompressionResult(BaseModel):
    """压缩结果数据模型"""

    original_size: int = Field(..., description="原始数据大小（字节）", ge=0)
    compressed_size: int = Field(..., description="压缩后大小（字节）", ge=0)
    compression_ratio: float = Field(..., description="压缩比", ge=0)
    algorithm_type: str = Field(..., description="算法类型（symbolic/fractal/gzip）")
    original_hash: str = Field(..., description="原始数据 SHA-256 哈希值")
    compression_time: float = Field(..., description="压缩耗时（秒）", ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    @field_validator("algorithm_type")
    @classmethod
    def validate_algorithm_type(cls, v: str) -> str:
        """验证算法类型是否合法"""
        allowed_types = {
            # 通用分类
            "symbolic",
            "fractal",
            "noise",
            # 序列算法
            "fibonacci",
            "recursive",
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

    @field_validator("original_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """验证哈希值格式（SHA-256 为 64 个十六进制字符）"""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"SHA-256 哈希值必须是 64 个十六进制字符，当前值: {v}")
        return v.lower()


class AlgorithmDescriptor(BaseModel):
    """算法描述符数据模型"""

    algorithm_type: str = Field(..., description="算法类型")
    parameters: Dict[str, Any] = Field(..., description="算法参数")
    code: Optional[str] = Field(default=None, description="生成代码（如有）")
    version: str = Field(default="1.0", description="格式版本号")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """验证版本号格式（简单的语义化版本）"""
        parts = v.split(".")
        if len(parts) < 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"版本号格式错误，应为 'X.Y' 或 'X.Y.Z' 格式，当前值: {v}")
        return v


class PatternInfo(BaseModel):
    """模式信息数据模型"""

    pattern_type: str = Field(..., description="模式类型（sequence/fractal/noise）")
    confidence: float = Field(..., description="置信度（0-1）", ge=0, le=1)
    parameters: Dict[str, Any] = Field(..., description="模式参数")
    detected_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="检测时间（ISO 8601 格式）",
    )

    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        """验证模式类型是否合法"""
        allowed_types = {
            # General types
            "sequence",
            "fractal",
            "noise",
            "unknown",
            # Sequence subtypes
            "arithmetic",
            "geometric",
            "fibonacci",
            "recursive",
            "polynomial",
            "primes",
            "pi_digits",
            # Fractal subtypes
            "mandelbrot",
            "julia",
            # 2D pattern subtypes
            "checkerboard",
            "stripes",
            # Noise subtypes
            "perlin_noise",
        }
        if v not in allowed_types:
            raise ValueError(f"模式类型必须是 {allowed_types} 之一，当前值: {v}")
        return v


class VerificationResult(BaseModel):
    """验证结果数据模型"""

    is_valid: bool = Field(..., description="是否验证通过")
    original_hash: str = Field(..., description="原始数据哈希值")
    reconstructed_hash: str = Field(..., description="重建数据哈希值")
    error_message: Optional[str] = Field(default=None, description="错误信息（如有）")
    verification_time: float = Field(..., description="验证耗时（秒）", ge=0)

    @field_validator("original_hash", "reconstructed_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """验证哈希值格式（SHA-256 为 64 个十六进制字符）"""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"SHA-256 哈希值必须是 64 个十六进制字符，当前值: {v}")
        return v.lower()
