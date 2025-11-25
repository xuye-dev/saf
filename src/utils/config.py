"""
配置管理模块（支持 YAML 配置文件）

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """日志配置"""

    level: str = Field(default="INFO", description="日志级别")
    file_path: str = Field(default="logs/saf.log", description="日志文件路径")
    console: bool = Field(default=True, description="是否输出到控制台")
    max_file_size_mb: int = Field(default=10, description="单个日志文件最大大小（MB）", ge=1)
    backup_count: int = Field(default=3, description="日志文件备份数量", ge=0)


class PerformanceConfig(BaseModel):
    """性能配置"""

    max_memory_mb: int = Field(default=4096, description="最大内存使用（MB）", ge=128)
    enable_parallel: bool = Field(default=True, description="是否启用并行处理")
    worker_threads: int = Field(default=4, description="工作线程数", ge=1, le=64)


class CompressionConfig(BaseModel):
    """压缩配置"""

    fallback_to_gzip: bool = Field(default=True, description="无法算法压缩时是否回退到 gzip")
    verification_required: bool = Field(default=True, description="是否强制验证无损性")
    min_compression_ratio: float = Field(
        default=2.0,
        description="最小压缩比阈值（低于此值回退到 gzip）",
        ge=1.0,
    )


class DetectionConfig(BaseModel):
    """模式检测配置"""

    max_iterations: int = Field(default=10000, description="符号回归最大迭代次数", ge=100)
    confidence_threshold: float = Field(
        default=0.9, description="模式置信度阈值（0-1）", ge=0, le=1
    )
    timeout_seconds: int = Field(default=300, description="单个检测任务超时时间（秒）", ge=1)


class Config(BaseModel):
    """全局配置数据模型"""

    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="日志配置")
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="性能配置"
    )
    compression: CompressionConfig = Field(
        default_factory=CompressionConfig, description="压缩配置"
    )
    detection: DetectionConfig = Field(default_factory=DetectionConfig, description="模式检测配置")


class ConfigLoader:
    """配置加载器（单例模式）"""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Config] = None

    def __new__(cls) -> "ConfigLoader":
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[Path] = None) -> Config:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径，默认为 config/default.yaml

        Returns:
            Config: 配置对象

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 解析错误
            ValueError: 配置验证失败
        """
        if config_path is None:
            # 默认配置文件路径（项目根目录）
            config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 读取 YAML 文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 使用 Pydantic 验证并创建配置对象
        self._config = Config(**config_dict)
        return self._config

    def get(self) -> Config:
        """
        获取当前配置对象

        Returns:
            Config: 配置对象

        Raises:
            RuntimeError: 配置未加载
        """
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load() 方法加载配置文件")
        return self._config

    def reload(self, config_path: Optional[Path] = None) -> Config:
        """
        重新加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            Config: 配置对象
        """
        return self.load(config_path)


# 全局配置加载器实例
_config_loader = ConfigLoader()


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    加载配置文件（便捷函数）

    Args:
        config_path: 配置文件路径，默认为 config/default.yaml

    Returns:
        Config: 配置对象

    Example:
        >>> config = load_config()
        >>> print(config.logging.level)
        INFO
    """
    return _config_loader.load(config_path)


def get_config() -> Config:
    """
    获取当前配置对象（便捷函数）

    Returns:
        Config: 配置对象

    Raises:
        RuntimeError: 配置未加载

    Example:
        >>> config = get_config()
        >>> print(config.performance.worker_threads)
        4
    """
    return _config_loader.get()
