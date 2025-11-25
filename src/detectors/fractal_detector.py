"""
分形模式检测器（检测分形图像的生成参数）

Author: 徐野
Date: 2025-11-23
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..generators.fractal import JuliaGenerator, MandelbrotGenerator
from ..models import PatternInfo
from .base import BaseDetector, ProgressCallback, calculate_r_squared


class FractalDetector(BaseDetector):
    """
    分形模式检测器

    支持检测以下分形类型：
    1. Mandelbrot 集合（参数匹配法）
    2. Julia 集合（参数匹配法）

    检测策略：
    - 维护常见参数组合库
    - 对输入图像尝试参数匹配
    - 返回相似度最高的参数组合
    """

    # Mandelbrot 常见参数组合
    MANDELBROT_PRESETS: List[Dict[str, Any]] = [
        # 标准视图
        {
            "x_min": -2.5,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "max_iter": 256,
        },
        # 放大视图 1
        {
            "x_min": -0.8,
            "x_max": -0.4,
            "y_min": -0.2,
            "y_max": 0.2,
            "max_iter": 512,
        },
        # 放大视图 2（海马谷）
        {
            "x_min": -0.75,
            "x_max": -0.73,
            "y_min": 0.1,
            "y_max": 0.12,
            "max_iter": 1024,
        },
    ]

    # Julia 常见参数组合
    JULIA_PRESETS: List[Dict[str, Any]] = [
        # 经典 Julia 集
        {
            "c": -0.7 + 0.27015j,
            "x_min": -1.5,
            "x_max": 1.5,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iter": 256,
        },
        # 对称 Julia 集
        {
            "c": -0.4 + 0.6j,
            "x_min": -1.5,
            "x_max": 1.5,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iter": 256,
        },
        # 龙形 Julia 集
        {
            "c": -0.8 + 0.156j,
            "x_min": -1.5,
            "x_max": 1.5,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iter": 256,
        },
    ]

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        """
        初始化分形检测器

        Args:
            confidence_threshold: 置信度阈值（低于此值的模式将被拒绝）
        """
        super().__init__(name="FractalDetector")
        self.confidence_threshold = confidence_threshold
        self.update_metadata(detector_type="fractal")

    def detect(
        self,
        data: NDArray[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> PatternInfo:
        """
        检测分形图像的生成参数

        Args:
            data: 待检测的分形图像数据（2D NumPy 数组，迭代次数矩阵）
            progress_callback: 进度回调函数（可选）
            **kwargs: 额外参数（保留）

        Returns:
            PatternInfo: 检测到的模式信息

        Raises:
            ValueError: 数据格式不正确

        Example:
            >>> detector = FractalDetector()
            >>> gen = MandelbrotGenerator()
            >>> iterations = gen.generate(width=800, height=600)
            >>> pattern = detector.detect(iterations)
            >>> print(pattern.pattern_type)  # "mandelbrot"
            >>> print(pattern.confidence)     # > 0.95
        """
        if data.ndim != 2:
            raise ValueError(f"分形数据必须是二维数组，当前维度: {data.ndim}")

        height, width = data.shape

        progress = ProgressCallback(progress_callback)

        # 优先检测简单图案（棋盘、条纹），因为这些检测更快
        progress(0.05)
        checkerboard_result = self._detect_checkerboard(data, width, height)
        if checkerboard_result.confidence >= self.confidence_threshold:
            progress(1.0)
            return checkerboard_result

        progress(0.1)
        stripes_result = self._detect_stripes(data, width, height)
        if stripes_result.confidence >= self.confidence_threshold:
            progress(1.0)
            return stripes_result

        # 尝试 Perlin 噪声检测（针对 float 类型数据）
        progress(0.2)
        if np.issubdtype(data.dtype, np.floating):
            perlin_result = self._detect_perlin_noise(data, width, height)
            if perlin_result.confidence >= self.confidence_threshold:
                progress(1.0)
                return perlin_result

        # 再尝试 Mandelbrot 检测
        progress(0.3)
        mandelbrot_result = self._detect_mandelbrot(data, width, height)
        progress(0.6)

        # 再尝试 Julia 检测
        julia_result = self._detect_julia(data, width, height)
        progress(0.9)

        # 选择置信度更高的结果
        if mandelbrot_result.confidence > julia_result.confidence:
            best_result = mandelbrot_result
        else:
            best_result = julia_result

        progress(1.0)

        # 如果置信度低于阈值，返回 unknown
        if best_result.confidence < self.confidence_threshold:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        return best_result

    def _detect_mandelbrot(
        self, data: NDArray[Any], width: int, height: int
    ) -> PatternInfo:
        """
        检测 Mandelbrot 分形参数

        Args:
            data: 分形图像数据
            width: 图像宽度
            height: 图像高度

        Returns:
            PatternInfo: 检测结果
        """
        best_confidence = 0.0
        best_params: Optional[Dict[str, Any]] = None

        # 从数据中推断 max_iter（分形数据的最大值 + 1）
        inferred_max_iter = int(data.max()) + 1

        gen = MandelbrotGenerator()

        # 遍历所有预设参数
        for preset in self.MANDELBROT_PRESETS:
            # 生成候选分形（使用推断的 max_iter）
            try:
                candidate = gen.generate(
                    width=width,
                    height=height,
                    max_iter=inferred_max_iter,  # 使用推断的 max_iter
                    x_min=preset["x_min"],
                    x_max=preset["x_max"],
                    y_min=preset["y_min"],
                    y_max=preset["y_max"],
                )

                # 计算相似度（使用归一化 R²）
                similarity = self._calculate_similarity(data, candidate)

                # 更新最佳匹配
                if similarity > best_confidence:
                    best_confidence = similarity
                    # 使用推断的 max_iter，而不是 preset 中的值
                    best_params = preset.copy()
                    best_params["max_iter"] = inferred_max_iter

            except Exception:
                continue

        if best_params is None or best_confidence < 0.85:
            return PatternInfo(
                pattern_type="mandelbrot",
                confidence=0.0,
                parameters={},
            )

        # 添加 width 和 height 参数（用于重建数据）
        best_params["width"] = width
        best_params["height"] = height

        return PatternInfo(
            pattern_type="mandelbrot",
            confidence=float(best_confidence),
            parameters=best_params,
        )

    def _detect_julia(self, data: NDArray[Any], width: int, height: int) -> PatternInfo:
        """
        检测 Julia 分形参数

        Args:
            data: 分形图像数据
            width: 图像宽度
            height: 图像高度

        Returns:
            PatternInfo: 检测结果
        """
        best_confidence = 0.0
        best_params: Optional[Dict[str, Any]] = None

        # 从数据中推断 max_iter（分形数据的最大值 + 1）
        inferred_max_iter = int(data.max()) + 1

        # 遍历所有预设参数
        for preset in self.JULIA_PRESETS:
            try:
                gen = JuliaGenerator(c=preset["c"])
                candidate = gen.generate(
                    width=width,
                    height=height,
                    max_iter=inferred_max_iter,  # 使用推断的 max_iter
                    x_min=preset["x_min"],
                    x_max=preset["x_max"],
                    y_min=preset["y_min"],
                    y_max=preset["y_max"],
                )

                # 计算相似度
                similarity = self._calculate_similarity(data, candidate)

                # 更新最佳匹配
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_params = {
                        "c_real": preset["c"].real,
                        "c_imag": preset["c"].imag,
                        "x_min": preset["x_min"],
                        "x_max": preset["x_max"],
                        "y_min": preset["y_min"],
                        "y_max": preset["y_max"],
                        "max_iter": inferred_max_iter,  # 使用推断的 max_iter
                    }

            except Exception:
                continue

        if best_params is None or best_confidence < 0.85:
            return PatternInfo(
                pattern_type="julia",
                confidence=0.0,
                parameters={},
            )

        # 添加 width 和 height 参数（用于重建数据）
        best_params["width"] = width
        best_params["height"] = height

        return PatternInfo(
            pattern_type="julia",
            confidence=float(best_confidence),
            parameters=best_params,
        )

    def _calculate_similarity(
        self, data1: NDArray[Any], data2: NDArray[Any]
    ) -> float:
        """
        计算两个分形图像的相似度

        Args:
            data1: 第一个图像（迭代次数矩阵）
            data2: 第二个图像（迭代次数矩阵）

        Returns:
            float: 相似度（0-1 之间）

        策略：
        1. 归一化到 [0, 1]
        2. 计算像素级别的 R²
        """
        if data1.shape != data2.shape:
            return 0.0

        # 归一化到 [0, 1]
        data1_norm = data1.astype(np.float64)
        data2_norm = data2.astype(np.float64)

        if data1_norm.max() > 0:
            data1_norm = data1_norm / data1_norm.max()
        if data2_norm.max() > 0:
            data2_norm = data2_norm / data2_norm.max()

        # 展平为一维数组
        data1_flat = data1_norm.flatten()
        data2_flat = data2_norm.flatten()

        # 计算 R²
        r_squared = calculate_r_squared(data1_flat, data2_flat)

        return r_squared

    def _detect_checkerboard(
        self, data: NDArray[Any], width: int, height: int
    ) -> PatternInfo:
        """
        检测棋盘图案

        Args:
            data: 图像数据
            width: 图像宽度
            height: 图像高度

        Returns:
            PatternInfo: 检测结果
        """
        # 获取唯一值
        unique_values = np.unique(data)

        # 棋盘图案应该只有2个不同的值
        if len(unique_values) != 2:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        color_0, color_1 = sorted(unique_values)

        # 尝试不同的格子大小（从1到min(width, height)//2）
        best_confidence = 0.0
        best_cell_size = 1
        max_cell_size = min(width, height) // 2

        for cell_size in range(1, max_cell_size + 1):
            # 生成候选棋盘图案
            candidate = np.zeros((height, width), dtype=data.dtype)
            for i in range(height):
                for j in range(width):
                    if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                        candidate[i, j] = color_0
                    else:
                        candidate[i, j] = color_1

            # 计算匹配度
            matches = np.sum(data == candidate)
            confidence = matches / (width * height)

            if confidence > best_confidence:
                best_confidence = confidence
                best_cell_size = cell_size

            # 如果找到完美匹配，提前退出
            if confidence >= 0.9999:
                break

        # 只有当匹配度非常高时才认为是棋盘图案
        if best_confidence < 0.999:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        return PatternInfo(
            pattern_type="checkerboard",
            confidence=float(best_confidence),
            parameters={
                "height": height,
                "width": width,
                "cell_size": best_cell_size,
                "color_0": int(color_0),
                "color_1": int(color_1),
            },
        )

    def _detect_stripes(
        self, data: NDArray[Any], width: int, height: int
    ) -> PatternInfo:
        """
        检测条纹图案（水平或垂直）

        Args:
            data: 图像数据
            width: 图像宽度
            height: 图像高度

        Returns:
            PatternInfo: 检测结果
        """
        # 获取唯一值
        unique_values = np.unique(data)

        # 条纹图案应该只有2个不同的值
        if len(unique_values) != 2:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        color_0, color_1 = sorted(unique_values)

        # 尝试水平条纹
        best_confidence = 0.0
        best_params: Dict[str, Any] = {}

        max_stripe_width = min(width, height) // 2

        for stripe_width in range(1, max_stripe_width + 1):
            # 水平条纹
            candidate_h = np.zeros((height, width), dtype=data.dtype)
            for i in range(height):
                if (i // stripe_width) % 2 == 0:
                    candidate_h[i, :] = color_0
                else:
                    candidate_h[i, :] = color_1

            matches_h = np.sum(data == candidate_h)
            confidence_h = matches_h / (width * height)

            if confidence_h > best_confidence:
                best_confidence = confidence_h
                best_params = {
                    "height": height,
                    "width": width,
                    "direction": "horizontal",
                    "stripe_width": stripe_width,
                    "color_0": int(color_0),
                    "color_1": int(color_1),
                }

            # 垂直条纹
            candidate_v = np.zeros((height, width), dtype=data.dtype)
            for j in range(width):
                if (j // stripe_width) % 2 == 0:
                    candidate_v[:, j] = color_0
                else:
                    candidate_v[:, j] = color_1

            matches_v = np.sum(data == candidate_v)
            confidence_v = matches_v / (width * height)

            if confidence_v > best_confidence:
                best_confidence = confidence_v
                best_params = {
                    "height": height,
                    "width": width,
                    "direction": "vertical",
                    "stripe_width": stripe_width,
                    "color_0": int(color_0),
                    "color_1": int(color_1),
                }

            # 如果找到完美匹配，提前退出
            if best_confidence >= 0.9999:
                break

        # 只有当匹配度非常高时才认为是条纹图案
        if best_confidence < 0.999:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        return PatternInfo(
            pattern_type="stripes",
            confidence=float(best_confidence),
            parameters=best_params,
        )

    def _detect_perlin_noise(
        self, data: NDArray[Any], width: int, height: int
    ) -> PatternInfo:
        """
        检测 Perlin 噪声参数

        Args:
            data: 噪声数据
            width: 图像宽度
            height: 图像高度

        Returns:
            PatternInfo: 检测结果
        """
        # Perlin 噪声的特征：
        # 1. 数据类型通常为 float
        # 2. 值域通常在 [-1, 1] 范围内
        # 3. 具有平滑的频率特征

        # 检查数据类型
        if not np.issubdtype(data.dtype, np.floating):
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        # 检查值域（Perlin 噪声通常在 [-1, 1] 范围内）
        data_min = float(data.min())
        data_max = float(data.max())

        if data_min < -1.5 or data_max > 1.5:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        # 常见的 Perlin 噪声参数组合
        scale_candidates = [50.0, 100.0, 150.0, 200.0]
        octaves_candidates = [4, 6, 8]
        persistence_candidates = [0.5, 0.6]
        lacunarity_candidates = [2.0, 2.5]

        # 搜索 seed（在有限范围内）
        seed_candidates = list(range(0, 100, 10)) + list(range(100, 1000, 100)) + [42]

        best_confidence = 0.0
        best_params: Optional[Dict[str, Any]] = None

        try:
            import noise as noise_lib  # type: ignore
        except ImportError:
            # 如果没有 noise 库，无法检测
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        # 搜索最佳参数组合（采样检测，避免完整重建）
        for scale in scale_candidates:
            for octaves in octaves_candidates:
                for persistence in persistence_candidates:
                    for lacunarity in lacunarity_candidates:
                        for seed in seed_candidates:
                            # 只采样部分点进行匹配（减少计算量）
                            sample_points = 100
                            sample_y = np.random.RandomState(0).randint(0, height, sample_points)
                            sample_x = np.random.RandomState(1).randint(0, width, sample_points)

                            # 生成采样点的噪声值
                            sample_noise = np.array([
                                noise_lib.pnoise2(
                                    x / scale,
                                    y / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=width,
                                    repeaty=height,
                                    base=seed,
                                )
                                for y, x in zip(sample_y, sample_x)
                            ])

                            # 计算相似度（使用均方误差）
                            sample_data = data[sample_y, sample_x]
                            mse = np.mean((sample_data - sample_noise) ** 2)

                            # 转换为置信度（MSE 越小，置信度越高）
                            # 对于 Perlin 噪声，MSE < 0.001 认为是很好的匹配
                            confidence = np.exp(-mse * 1000)

                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_params = {
                                    "width": width,
                                    "height": height,
                                    "scale": scale,
                                    "octaves": octaves,
                                    "persistence": persistence,
                                    "lacunarity": lacunarity,
                                    "seed": seed,
                                }

                            # 如果找到高置信度匹配，提前退出
                            if confidence >= 0.99:
                                return PatternInfo(
                                    pattern_type="perlin_noise",
                                    confidence=float(confidence),
                                    parameters=best_params,
                                )

        if best_confidence < 0.85 or best_params is None:
            return PatternInfo(
                pattern_type="unknown",
                confidence=0.0,
                parameters={},
            )

        return PatternInfo(
            pattern_type="perlin_noise",
            confidence=float(best_confidence),
            parameters=best_params,
        )
