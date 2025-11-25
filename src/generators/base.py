"""
数据生成器抽象基类

Author: 徐野
Date: 2025-11-23
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray


class BaseGenerator(ABC):
    """
    数据生成器抽象基类

    所有生成器（序列、分形、噪声）都应继承此基类并实现 generate() 方法。
    """

    def __init__(self, name: str) -> None:
        """
        初始化生成器

        Args:
            name: 生成器名称（用于日志和元数据）
        """
        self.name = name
        self._metadata: Dict[str, Any] = {"generator_name": name}

    @abstractmethod
    def generate(self, **kwargs: Any) -> NDArray[Any]:
        """
        生成数据（抽象方法，子类必须实现）

        Args:
            **kwargs: 生成器参数（具体参数由子类定义）

        Returns:
            np.ndarray: 生成的数据数组

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError("子类必须实现 generate() 方法")

    def save_npy(self, data: NDArray[Any], output_path: Union[str, Path]) -> None:
        """
        保存数据为 NumPy 二进制格式 (.npy)

        Args:
            data: 要保存的数据数组
            output_path: 输出文件路径

        Raises:
            IOError: 文件写入失败

        Example:
            >>> generator = SomeGenerator("test")
            >>> data = generator.generate(n=100)
            >>> generator.save_npy(data, "output.npy")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, data)

    def save_txt(
        self, data: NDArray[Any], output_path: Union[str, Path], fmt: str = "%.18e"
    ) -> None:
        """
        保存数据为文本格式 (.txt)

        Args:
            data: 要保存的数据数组
            output_path: 输出文件路径
            fmt: 数字格式（默认科学计数法，18 位精度）

        Raises:
            IOError: 文件写入失败

        Example:
            >>> generator.save_txt(data, "output.txt", fmt="%.6f")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, data, fmt=fmt)

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取生成器元数据

        Returns:
            Dict[str, Any]: 元数据字典（包含生成器名称和其他信息）

        Example:
            >>> metadata = generator.get_metadata()
            >>> print(metadata["generator_name"])
        """
        return self._metadata.copy()

    def update_metadata(self, **kwargs: Any) -> None:
        """
        更新元数据

        Args:
            **kwargs: 要更新的元数据键值对

        Example:
            >>> generator.update_metadata(data_type="sequence", length=1000)
        """
        self._metadata.update(kwargs)


class ProgressCallback:
    """进度回调辅助类（用于耗时操作的进度提示）"""

    def __init__(self, callback: Optional[Callable[[float], None]] = None) -> None:
        """
        初始化进度回调

        Args:
            callback: 回调函数，接收 0-1 之间的进度值
        """
        self.callback = callback

    def update(self, progress: float) -> None:
        """
        更新进度

        Args:
            progress: 进度值（0-1）
        """
        if self.callback is not None:
            self.callback(max(0.0, min(1.0, progress)))

    def __call__(self, progress: float) -> None:
        """允许直接调用实例来更新进度"""
        self.update(progress)
