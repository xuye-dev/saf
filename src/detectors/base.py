"""
模式检测器抽象基类

Author: 徐野
Date: 2025-11-23
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..models import PatternInfo


class BaseDetector(ABC):
    """
    模式检测器抽象基类

    所有检测器（序列检测器、分形检测器等）都应继承此基类并实现 detect() 方法。
    """

    def __init__(self, name: str, timeout: float = 30.0) -> None:
        """
        初始化检测器

        Args:
            name: 检测器名称（用于日志和元数据）
            timeout: 检测超时时间（秒），默认 30 秒
        """
        self.name = name
        self.timeout = timeout
        self._metadata: Dict[str, Any] = {"detector_name": name}

    @abstractmethod
    def detect(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        检测数据的生成模式（抽象方法，子类必须实现）

        Args:
            data: 待检测的数据（NumPy 数组）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外的检测参数（具体参数由子类定义）

        Returns:
            PatternInfo: 检测到的模式信息

        Raises:
            NotImplementedError: 子类未实现此方法
            TimeoutError: 检测超时
        """
        raise NotImplementedError("子类必须实现 detect() 方法")

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取检测器元数据

        Returns:
            Dict[str, Any]: 元数据字典（包含检测器名称和其他信息）

        Example:
            >>> metadata = detector.get_metadata()
            >>> print(metadata["detector_name"])
        """
        return self._metadata.copy()

    def update_metadata(self, **kwargs: Any) -> None:
        """
        更新元数据

        Args:
            **kwargs: 要更新的元数据键值对

        Example:
            >>> detector.update_metadata(pattern_type="sequence", confidence=0.95)
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


def calculate_r_squared(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """
    计算 R² 决定系数（拟合优度）

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        float: R² 值（0-1 之间，越接近 1 表示拟合越好）

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> r2 = calculate_r_squared(y_true, y_pred)
        >>> print(r2)  # 接近 1.0
    """
    # 检查预测值是否包含 nan 或 inf（拟合失败的标志）
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return 0.0

    # 计算残差平方和（SS_res）
    ss_res = np.sum((y_true - y_pred) ** 2)

    # 计算总平方和（SS_tot）
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # 计算 R²
    if ss_tot == 0:
        return 0.0
    r_squared = 1 - (ss_res / ss_tot)

    # 检查 R² 结果是否为 nan 或 inf
    if np.isnan(r_squared) or np.isinf(r_squared):
        return 0.0

    # R² 可能为负（拟合比均值还差），截断到 [0, 1]
    return float(max(0.0, min(1.0, r_squared)))


def calculate_relative_error(true_value: float, predicted_value: float) -> float:
    """
    计算相对误差

    Args:
        true_value: 真实值
        predicted_value: 预测值

    Returns:
        float: 相对误差（百分比，0-1 之间）

    Example:
        >>> error = calculate_relative_error(100.0, 95.0)
        >>> print(error)  # 0.05 (5%)
    """
    if true_value == 0:
        return 0.0 if predicted_value == 0 else 1.0

    return abs(true_value - predicted_value) / abs(true_value)
