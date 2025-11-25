"""
通用模式匹配器（检测器编排器）

Author: 徐野
Date: 2025-11-23
"""

from typing import Any, Callable, Optional

from numpy.typing import NDArray

from ..models import PatternInfo
from .base import BaseDetector, ProgressCallback
from .fractal_detector import FractalDetector
from .sequence_detector import SequenceDetector


class PatternMatcher(BaseDetector):
    """
    通用模式匹配器

    根据数据类型自动选择合适的检测器：
    - 1D 数据 → SequenceDetector
    - 2D 数据 → FractalDetector

    提供统一的检测接口，简化使用流程。
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        """
        初始化模式匹配器

        Args:
            confidence_threshold: 置信度阈值（低于此值的模式将被拒绝）
        """
        super().__init__(name="PatternMatcher")
        self.confidence_threshold = confidence_threshold
        self.update_metadata(detector_type="universal")

        # 初始化各类检测器
        self.sequence_detector = SequenceDetector(
            confidence_threshold=confidence_threshold
        )
        self.fractal_detector = FractalDetector(confidence_threshold=confidence_threshold)

    def detect(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        自动检测数据的生成模式

        Args:
            data: 待检测的数据（NumPy 数组，支持 1D 或 2D）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外参数（传递给具体检测器）

        Returns:
            PatternInfo: 检测到的模式信息

        Raises:
            ValueError: 数据格式不正确（维度 > 2）

        Example:
            >>> matcher = PatternMatcher()
            >>> # 检测序列
            >>> seq = np.array([0, 1, 1, 2, 3, 5, 8, 13])
            >>> pattern = matcher.detect(seq)
            >>> print(pattern.pattern_type)  # "fibonacci"
            >>>
            >>> # 检测分形
            >>> gen = MandelbrotGenerator()
            >>> fractal = gen.generate(width=800, height=600)
            >>> pattern = matcher.detect(fractal)
            >>> print(pattern.pattern_type)  # "mandelbrot"
        """
        progress = ProgressCallback(progress_callback)

        # 根据数据维度选择检测器
        if data.ndim == 1:
            # 一维数据 → 序列检测器
            progress(0.1)
            result = self.sequence_detector.detect(
                data, progress_callback=lambda p: progress(0.1 + 0.9 * p), **kwargs
            )

        elif data.ndim == 2:
            # 二维数据 → 分形检测器
            progress(0.1)
            result = self.fractal_detector.detect(
                data, progress_callback=lambda p: progress(0.1 + 0.9 * p), **kwargs
            )

        else:
            raise ValueError(
                f"不支持的数据维度: {data.ndim}（仅支持 1D 序列或 2D 图像）"
            )

        progress(1.0)
        return result

    def detect_sequence(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        显式调用序列检测器

        Args:
            data: 待检测的序列数据（1D NumPy 数组）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外参数

        Returns:
            PatternInfo: 检测到的模式信息
        """
        return self.sequence_detector.detect(data, progress_callback, **kwargs)

    def detect_fractal(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        显式调用分形检测器

        Args:
            data: 待检测的分形图像数据（2D NumPy 数组）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外参数

        Returns:
            PatternInfo: 检测到的模式信息
        """
        return self.fractal_detector.detect(data, progress_callback, **kwargs)
