"""
分形图像生成器（Mandelbrot 集、Julia 集）

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import BaseGenerator, ProgressCallback


class MandelbrotGenerator(BaseGenerator):
    """Mandelbrot 集合生成器"""

    def __init__(self) -> None:
        """初始化 Mandelbrot 生成器"""
        super().__init__(name="Mandelbrot")
        self.update_metadata(data_type="fractal", pattern_type="mandelbrot")

    def generate(
        self,
        width: int = 800,
        height: int = 600,
        max_iter: int = 256,
        x_min: float = -2.5,
        x_max: float = 1.0,
        y_min: float = -1.0,
        y_max: float = 1.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成 Mandelbrot 集合的迭代次数矩阵

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            max_iter: 最大迭代次数
            x_min: 复平面实部最小值
            x_max: 复平面实部最大值
            y_min: 复平面虚部最小值
            y_max: 复平面虚部最大值
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 迭代次数矩阵 (height, width)，dtype=int32

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = MandelbrotGenerator()
            >>> iterations = gen.generate(width=800, height=600, max_iter=256)
            >>> print(iterations.shape)
            (600, 800)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")
        if max_iter < 1:
            raise ValueError(f"最大迭代次数必须 >= 1，当前值: {max_iter}")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("复平面范围不合法")

        progress = ProgressCallback(progress_callback)

        # 创建复平面网格
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        # 初始化迭代次数矩阵
        iterations = np.zeros((height, width), dtype=np.int32)
        Z = np.zeros_like(C, dtype=np.complex128)

        # 使用向量化计算 Mandelbrot 集合
        for i in range(max_iter):
            # 计算 Z = Z^2 + C
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            iterations[mask] = i

            # 更新进度
            if i % 10 == 0:
                progress(i / max_iter)

        progress(1.0)
        self.update_metadata(
            width=width,
            height=height,
            max_iter=max_iter,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )
        return iterations

    def save_png(
        self,
        iterations: NDArray[Any],
        output_path: Union[str, Path],
        colormap: str = "hot",
    ) -> None:
        """
        保存迭代次数矩阵为 PNG 图像

        Args:
            iterations: 迭代次数矩阵
            output_path: 输出文件路径
            colormap: 颜色映射方案（'hot', 'cool', 'gray', 'viridis'）

        Raises:
            IOError: 文件写入失败

        Example:
            >>> gen = MandelbrotGenerator()
            >>> iterations = gen.generate(width=800, height=600)
            >>> gen.save_png(iterations, "mandelbrot.png", colormap="hot")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 归一化到 0-255
        normalized = (iterations / iterations.max() * 255).astype(np.uint8)

        # 应用颜色映射
        if colormap == "hot":
            image = self._apply_hot_colormap(normalized)
        elif colormap == "cool":
            image = self._apply_cool_colormap(normalized)
        elif colormap == "viridis":
            image = self._apply_viridis_colormap(normalized)
        else:  # gray
            image = Image.fromarray(normalized, mode="L")

        image.save(output_path)

    def _apply_hot_colormap(self, data: NDArray[Any]) -> Image.Image:
        """应用 'hot' 颜色映射（黑-红-黄-白）"""
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.minimum(255, data * 3)  # Red
        rgb[..., 1] = np.maximum(0, np.minimum(255, (data - 85) * 3))  # Green
        rgb[..., 2] = np.maximum(0, data - 170)  # Blue
        return Image.fromarray(rgb, mode="RGB")

    def _apply_cool_colormap(self, data: NDArray[Any]) -> Image.Image:
        """应用 'cool' 颜色映射（青-品红）"""
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[..., 0] = data  # Red
        rgb[..., 1] = 255 - data  # Green (cyan to magenta)
        rgb[..., 2] = 255  # Blue
        return Image.fromarray(rgb, mode="RGB")

    def _apply_viridis_colormap(self, data: NDArray[Any]) -> Image.Image:
        """应用 'viridis' 颜色映射（简化版）"""
        # 简化的 viridis 近似（紫-蓝-绿-黄）
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        t = data / 255.0
        rgb[..., 0] = (255 * (0.267 + 0.005 * t + 0.322 * t**2)).astype(np.uint8)
        rgb[..., 1] = (255 * (0.004 + 0.873 * t - 0.331 * t**2)).astype(np.uint8)
        rgb[..., 2] = (255 * (0.329 - 0.183 * t + 0.152 * t**2)).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")


class JuliaGenerator(BaseGenerator):
    """Julia 集合生成器"""

    def __init__(self, c: complex = -0.7 + 0.27015j) -> None:
        """
        初始化 Julia 生成器

        Args:
            c: Julia 集合的复数常量（默认为经典值）

        Example:
            >>> gen = JuliaGenerator(c=-0.4 + 0.6j)
        """
        super().__init__(name="Julia")
        self.c = c
        self.update_metadata(
            data_type="fractal", pattern_type="julia", c_real=c.real, c_imag=c.imag
        )

    def generate(
        self,
        width: int = 800,
        height: int = 600,
        max_iter: int = 256,
        x_min: float = -1.5,
        x_max: float = 1.5,
        y_min: float = -1.5,
        y_max: float = 1.5,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成 Julia 集合的迭代次数矩阵

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            max_iter: 最大迭代次数
            x_min: 复平面实部最小值
            x_max: 复平面实部最大值
            y_min: 复平面虚部最小值
            y_max: 复平面虚部最大值
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 迭代次数矩阵 (height, width)，dtype=int32

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = JuliaGenerator(c=-0.7 + 0.27015j)
            >>> iterations = gen.generate(width=800, height=600, max_iter=256)
            >>> print(iterations.shape)
            (600, 800)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")
        if max_iter < 1:
            raise ValueError(f"最大迭代次数必须 >= 1，当前值: {max_iter}")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("复平面范围不合法")

        progress = ProgressCallback(progress_callback)

        # 创建复平面网格
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # 初始化迭代次数矩阵
        iterations = np.zeros((height, width), dtype=np.int32)

        # 使用向量化计算 Julia 集合
        for i in range(max_iter):
            # 计算 Z = Z^2 + c
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + self.c
            iterations[mask] = i

            # 更新进度
            if i % 10 == 0:
                progress(i / max_iter)

        progress(1.0)
        self.update_metadata(
            width=width,
            height=height,
            max_iter=max_iter,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )
        return iterations

    def save_png(
        self,
        iterations: NDArray[Any],
        output_path: Union[str, Path],
        colormap: str = "hot",
    ) -> None:
        """
        保存迭代次数矩阵为 PNG 图像

        Args:
            iterations: 迭代次数矩阵
            output_path: 输出文件路径
            colormap: 颜色映射方案（'hot', 'cool', 'gray', 'viridis'）

        Raises:
            IOError: 文件写入失败

        Example:
            >>> gen = JuliaGenerator(c=-0.4 + 0.6j)
            >>> iterations = gen.generate(width=800, height=600)
            >>> gen.save_png(iterations, "julia.png", colormap="viridis")
        """
        # 复用 MandelbrotGenerator 的 save_png 逻辑
        mandelbrot_gen = MandelbrotGenerator()
        mandelbrot_gen.save_png(iterations, output_path, colormap)
