"""
序列模式检测器（检测数学序列的生成规律）

Author: 徐野
Date: 2025-11-23
"""

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ..generators.sequence import PrimeGenerator
from ..models import PatternInfo
from .base import BaseDetector, ProgressCallback, calculate_r_squared


class SequenceDetector(BaseDetector):
    """
    序列模式检测器

    支持检测以下模式：
    1. 等差数列（线性规律）
    2. 等比数列（指数规律）
    3. 递归规律（如斐波那契数列）
    4. 多项式规律（如平方数、立方数）
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        """
        初始化序列检测器

        Args:
            confidence_threshold: 置信度阈值（低于此值的模式将被拒绝）
        """
        super().__init__(name="SequenceDetector")
        self.confidence_threshold = confidence_threshold
        self.update_metadata(detector_type="sequence")

    def detect(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        检测序列的生成模式

        Args:
            data: 待检测的序列数据（1D NumPy 数组）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外参数（保留）

        Returns:
            PatternInfo: 检测到的模式信息

        Raises:
            ValueError: 数据格式不正确

        Example:
            >>> detector = SequenceDetector()
            >>> data = np.array([0, 1, 1, 2, 3, 5, 8, 13])  # 斐波那契
            >>> pattern = detector.detect(data)
            >>> print(pattern.pattern_type)  # "fibonacci"
            >>> print(pattern.confidence)     # > 0.99
        """
        if data.ndim != 1:
            raise ValueError(f"序列数据必须是一维数组，当前维度: {data.ndim}")

        if len(data) < 3:
            raise ValueError(f"序列长度必须 >= 3，当前长度: {len(data)}")

        progress = ProgressCallback(progress_callback)

        # 依次尝试各种检测方法，按优先级从高到低
        detectors = [
            (self._detect_primes, 0.15),      # 素数序列检测（优先级最高）
            (self._detect_arithmetic, 0.3),   # 等差数列检测
            (self._detect_geometric, 0.45),   # 等比数列检测
            (self._detect_polynomial, 0.6),   # 多项式拟合检测（提前到递归之前）
            (self._detect_fibonacci, 0.8),    # 斐波那契/递归检测（移到多项式之后）
        ]

        best_pattern: Optional[PatternInfo] = None
        best_confidence = 0.0

        for i, (detector_func, progress_weight) in enumerate(detectors):
            pattern = detector_func(data)
            progress(progress_weight)

            # 更新最佳模式
            if pattern.confidence > best_confidence:
                best_confidence = pattern.confidence
                best_pattern = pattern

            # 如果置信度非常高（> 0.99），提前返回
            if pattern.confidence > 0.99:
                progress(1.0)
                return pattern

        progress(1.0)

        # 如果最佳模式的置信度低于阈值，返回 unknown
        if best_pattern is None or best_confidence < self.confidence_threshold:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        return best_pattern

    def _detect_primes(self, data: NDArray[Any]) -> PatternInfo:
        """
        检测素数序列

        Args:
            data: 序列数据

        Returns:
            PatternInfo: 检测结果
        """
        n = len(data)

        # 生成相同长度的素数序列进行比较
        try:
            prime_gen = PrimeGenerator()
            expected_primes = prime_gen.generate(n=n)

            # 完全匹配检测
            if np.array_equal(data, expected_primes):
                return PatternInfo(
                    pattern_type="primes",
                    confidence=1.0,
                    parameters={"n": n},
                )

            # 如果不完全匹配，返回低置信度
            return PatternInfo(pattern_type="primes", confidence=0.0, parameters={})

        except Exception:
            return PatternInfo(pattern_type="primes", confidence=0.0, parameters={})

    def _detect_arithmetic(self, data: NDArray[Any]) -> PatternInfo:
        """
        检测等差数列（a[n] = a[0] + n * d）

        Args:
            data: 序列数据

        Returns:
            PatternInfo: 检测结果
        """
        # 计算一阶差分
        diff = np.diff(data)

        # 检查差分是否恒定
        if len(diff) == 0:
            return PatternInfo(pattern_type="arithmetic", confidence=0.0, parameters={})

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        # 如果标准差非常小，说明是等差数列
        # 使用相对误差判断
        if mean_diff == 0:
            # 常数序列（特殊的等差数列，d=0）
            if std_diff == 0:
                return PatternInfo(
                    pattern_type="arithmetic",
                    confidence=1.0,
                    parameters={"first_term": float(data[0]), "common_difference": 0.0},
                )
            else:
                return PatternInfo(pattern_type="arithmetic", confidence=0.0, parameters={})

        relative_error = std_diff / abs(mean_diff)

        # 置信度计算：误差越小，置信度越高
        if relative_error < 1e-10:
            confidence = 1.0
        elif relative_error < 0.01:
            confidence = 0.99
        elif relative_error < 0.1:
            confidence = 0.90
        else:
            confidence = max(0.0, 1.0 - relative_error)

        return PatternInfo(
            pattern_type="arithmetic",
            confidence=float(confidence),
            parameters={
                "first_term": float(data[0]),
                "common_difference": float(mean_diff),
            },
        )

    def _detect_geometric(self, data: NDArray[Any]) -> PatternInfo:
        """
        检测等比数列（a[n] = a[0] * r^n）

        Args:
            data: 序列数据

        Returns:
            PatternInfo: 检测结果
        """
        # 避免除零
        if np.any(data[:-1] == 0):
            return PatternInfo(pattern_type="geometric", confidence=0.0, parameters={})

        # 计算相邻项的比值
        ratios = data[1:] / data[:-1]

        # 检查比值是否恒定
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        if mean_ratio == 0:
            return PatternInfo(pattern_type="geometric", confidence=0.0, parameters={})

        relative_error = std_ratio / abs(mean_ratio)

        # 置信度计算
        if relative_error < 1e-10:
            confidence = 1.0
        elif relative_error < 0.01:
            confidence = 0.99
        elif relative_error < 0.1:
            confidence = 0.90
        else:
            confidence = max(0.0, 1.0 - relative_error)

        return PatternInfo(
            pattern_type="geometric",
            confidence=float(confidence),
            parameters={
                "first_term": float(data[0]),
                "common_ratio": float(mean_ratio),
            },
        )

    def _detect_fibonacci(self, data: NDArray[Any]) -> PatternInfo:
        """
        检测斐波那契类递归规律（a[n] = k * a[n-1] + b * a[n-2]）

        Args:
            data: 序列数据

        Returns:
            PatternInfo: 检测结果
        """
        if len(data) < 3:
            return PatternInfo(pattern_type="fibonacci", confidence=0.0, parameters={})

        # 首先尝试精确整数递推检测（避免大数值导致的数值不稳定）
        # 检查是否满足 data[i] = k * data[i-1] + b * data[i-2] 的整数关系

        # 尝试标准斐波那契（k=1, b=1）
        is_standard_fib = True
        for i in range(2, len(data)):
            if data[i] != data[i - 1] + data[i - 2]:
                is_standard_fib = False
                break

        if is_standard_fib:
            return PatternInfo(
                pattern_type="fibonacci",
                confidence=1.0,
                parameters={
                    "first_term": float(data[0]),
                    "second_term": float(data[1]),
                    "coefficient_k": 1.0,
                    "coefficient_b": 1.0,
                },
            )

        # 尝试其他整数系数的递推（k=1, b=2 等常见形式）
        for k_int, b_int in [(1, 2), (2, 1), (1, 3), (2, 2)]:
            is_match = True
            for i in range(2, len(data)):
                if data[i] != k_int * data[i - 1] + b_int * data[i - 2]:
                    is_match = False
                    break
            if is_match:
                return PatternInfo(
                    pattern_type="recursive",
                    confidence=0.99,
                    parameters={
                        "first_term": float(data[0]),
                        "second_term": float(data[1]),
                        "coefficient_k": float(k_int),
                        "coefficient_b": float(b_int),
                    },
                )

        # 如果精确匹配失败，回退到最小二乘法（仅使用前 20 项避免数值不稳定）
        n_fit = min(20, len(data))
        data_fit = data[:n_fit].astype(np.float64)

        X = np.column_stack([data_fit[1:-1], data_fit[:-2]])
        y = data_fit[2:]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            k, b = coeffs

            # 重建完整序列验证
            reconstructed = np.zeros_like(data, dtype=np.float64)
            reconstructed[0] = data[0]
            reconstructed[1] = data[1]
            for i in range(2, len(data)):
                reconstructed[i] = k * reconstructed[i - 1] + b * reconstructed[i - 2]

            # 检查重建结果是否包含 nan 或 inf（数值溢出）
            if np.isnan(reconstructed).any() or np.isinf(reconstructed).any():
                return PatternInfo(pattern_type="fibonacci", confidence=0.0, parameters={})

            # 计算 R²
            r_squared = calculate_r_squared(data.astype(np.float64), reconstructed)

            if r_squared > 0.85:
                return PatternInfo(
                    pattern_type="recursive",
                    confidence=float(r_squared),
                    parameters={
                        "first_term": float(data[0]),
                        "second_term": float(data[1]),
                        "coefficient_k": float(k),
                        "coefficient_b": float(b),
                    },
                )

        except np.linalg.LinAlgError:
            pass

        return PatternInfo(pattern_type="fibonacci", confidence=0.0, parameters={})

    def _detect_polynomial(self, data: NDArray[Any]) -> PatternInfo:
        """
        检测多项式规律（a[n] = c0 + c1*n + c2*n^2 + ...）

        Args:
            data: 序列数据

        Returns:
            PatternInfo: 检测结果
        """
        n = len(data)
        indices = np.arange(n)

        # 尝试不同阶数的多项式拟合（1-4 阶）
        best_degree = 1
        best_r_squared = 0.0
        best_coeffs: Optional[NDArray[Any]] = None

        for degree in range(1, 5):
            try:
                # 多项式拟合
                coeffs = np.polyfit(indices, data, degree)
                poly = np.poly1d(coeffs)
                y_pred = poly(indices)

                # 计算 R²
                r_squared = calculate_r_squared(data.astype(np.float64), y_pred)

                # 更新最佳结果
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_degree = degree
                    best_coeffs = coeffs

            except (np.linalg.LinAlgError, np.exceptions.RankWarning):
                continue

        # 如果 R² 过低，拒绝
        if best_r_squared < 0.85 or best_coeffs is None:
            return PatternInfo(pattern_type="polynomial", confidence=0.0, parameters={})

        # np.polyfit 返回系数从高阶到低阶 [c_n, ..., c_1, c_0]
        # executor 期望从低阶到高阶 [c_0, c_1, ..., c_n]，需要反转
        reversed_coeffs = list(reversed(best_coeffs))

        return PatternInfo(
            pattern_type="polynomial",
            confidence=float(best_r_squared),
            parameters={
                "degree": int(best_degree),
                "coefficients": [float(c) for c in reversed_coeffs],
            },
        )
