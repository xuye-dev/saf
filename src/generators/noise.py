"""
噪声和图案生成器（Perlin 噪声、白噪声、棋盘、条纹）

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import noise
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import BaseGenerator, ProgressCallback


class PerlinNoiseGenerator(BaseGenerator):
    """Perlin 噪声生成器（使用 noise 库）"""

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        初始化 Perlin 噪声生成器

        Args:
            seed: 随机种子（可选），用于可重现的结果
        """
        super().__init__(name="PerlinNoise")
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.update_metadata(data_type="noise", pattern_type="perlin", seed=self.seed)

    def generate(
        self,
        width: int = 512,
        height: int = 512,
        scale: float = 100.0,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成 Perlin 噪声

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            scale: 噪声缩放比例（越大越平滑）
            octaves: 噪声倍频数（层次数）
            persistence: 振幅衰减系数（0-1）
            lacunarity: 频率增长系数（通常为 2.0）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: Perlin 噪声矩阵 (height, width)，值域 [-1, 1]

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = PerlinNoiseGenerator(seed=42)
            >>> noise_data = gen.generate(width=512, height=512, scale=100.0)
            >>> print(noise_data.shape)
            (512, 512)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")
        if scale <= 0:
            raise ValueError(f"scale 必须 > 0，当前值: {scale}")
        if octaves < 1:
            raise ValueError(f"octaves 必须 >= 1，当前值: {octaves}")

        progress = ProgressCallback(progress_callback)

        # 生成 Perlin 噪声
        noise_map = np.zeros((height, width), dtype=np.float64)

        for y in range(height):
            for x in range(width):
                noise_map[y, x] = noise.pnoise2(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=width,
                    repeaty=height,
                    base=self.seed,
                )

            # 更新进度（每行更新一次）
            if y % 10 == 0:
                progress(y / height)

        progress(1.0)
        self.update_metadata(
            width=width,
            height=height,
            scale=scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
        )
        return noise_map

    def save_png(
        self,
        noise_data: NDArray[Any],
        output_path: Union[str, Path],
        normalize: bool = True,
    ) -> None:
        """
        保存 Perlin 噪声为 PNG 图像

        Args:
            noise_data: 噪声数据矩阵
            output_path: 输出文件路径
            normalize: 是否归一化到 [0, 255]，默认 True

        Raises:
            IOError: 文件写入失败
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if normalize:
            # 归一化到 0-255
            normalized = (
                (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min()) * 255
            ).astype(np.uint8)
        else:
            normalized = noise_data.astype(np.uint8)

        image = Image.fromarray(normalized, mode="L")
        image.save(output_path)


class WhiteNoiseGenerator(BaseGenerator):
    """白噪声生成器（均匀分布随机噪声）"""

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        初始化白噪声生成器

        Args:
            seed: 随机种子（可选），用于可重现的结果
        """
        super().__init__(name="WhiteNoise")
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.update_metadata(data_type="noise", pattern_type="white", seed=seed)

    def generate(
        self,
        width: int = 512,
        height: int = 512,
        distribution: str = "uniform",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成白噪声

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            distribution: 分布类型（'uniform' 或 'normal'）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 白噪声矩阵 (height, width)

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = WhiteNoiseGenerator(seed=42)
            >>> noise_data = gen.generate(width=512, height=512, distribution="uniform")
            >>> print(noise_data.shape)
            (512, 512)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")

        progress = ProgressCallback(progress_callback)
        progress(0.5)

        if distribution == "uniform":
            # 均匀分布 [0, 1]
            noise_map = self.rng.rand(height, width)
        elif distribution == "normal":
            # 正态分布（均值 0，标准差 1）
            noise_map = self.rng.randn(height, width)
        else:
            raise ValueError(f"不支持的分布类型: {distribution}，必须为 'uniform' 或 'normal'")

        progress(1.0)
        self.update_metadata(width=width, height=height, distribution=distribution)
        return noise_map

    def save_png(
        self,
        noise_data: NDArray[Any],
        output_path: Union[str, Path],
        normalize: bool = True,
    ) -> None:
        """
        保存白噪声为 PNG 图像

        Args:
            noise_data: 噪声数据矩阵
            output_path: 输出文件路径
            normalize: 是否归一化到 [0, 255]，默认 True

        Raises:
            IOError: 文件写入失败
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if normalize:
            # 归一化到 0-255
            normalized = (
                (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min()) * 255
            ).astype(np.uint8)
        else:
            normalized = (np.clip(noise_data, 0, 1) * 255).astype(np.uint8)

        image = Image.fromarray(normalized, mode="L")
        image.save(output_path)


class CheckerboardGenerator(BaseGenerator):
    """棋盘图案生成器"""

    def __init__(self) -> None:
        """初始化棋盘生成器"""
        super().__init__(name="Checkerboard")
        self.update_metadata(data_type="pattern", pattern_type="checkerboard")

    def generate(
        self,
        width: int = 512,
        height: int = 512,
        square_size: int = 64,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成棋盘图案

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            square_size: 每个方格的边长（像素）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 棋盘图案矩阵 (height, width)，值为 0 或 1

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = CheckerboardGenerator()
            >>> pattern = gen.generate(width=512, height=512, square_size=64)
            >>> print(pattern.shape)
            (512, 512)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")
        if square_size < 1:
            raise ValueError(f"square_size 必须 >= 1，当前值: {square_size}")

        progress = ProgressCallback(progress_callback)
        progress(0.5)

        # 生成棋盘图案
        y_indices = np.arange(height) // square_size
        x_indices = np.arange(width) // square_size
        Y, X = np.meshgrid(y_indices, x_indices, indexing="ij")
        checkerboard = (X + Y) % 2

        progress(1.0)
        self.update_metadata(width=width, height=height, square_size=square_size)
        return cast(NDArray[Any], checkerboard.astype(np.int32))

    def save_png(self, pattern: NDArray[Any], output_path: Union[str, Path]) -> None:
        """
        保存棋盘图案为 PNG 图像

        Args:
            pattern: 棋盘图案矩阵
            output_path: 输出文件路径

        Raises:
            IOError: 文件写入失败
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为 0-255
        image_data = (pattern * 255).astype(np.uint8)
        image = Image.fromarray(image_data, mode="L")
        image.save(output_path)


class StripeGenerator(BaseGenerator):
    """条纹图案生成器"""

    def __init__(self) -> None:
        """初始化条纹生成器"""
        super().__init__(name="Stripe")
        self.update_metadata(data_type="pattern", pattern_type="stripe")

    def generate(
        self,
        width: int = 512,
        height: int = 512,
        stripe_width: int = 32,
        orientation: str = "horizontal",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成条纹图案

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            stripe_width: 条纹宽度（像素）
            orientation: 条纹方向（'horizontal' 或 'vertical'）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 条纹图案矩阵 (height, width)，值为 0 或 1

        Raises:
            ValueError: 参数不合法

        Example:
            >>> gen = StripeGenerator()
            >>> pattern = gen.generate(
            ...     width=512, height=512, stripe_width=32, orientation="horizontal"
            ... )
            >>> print(pattern.shape)
            (512, 512)
        """
        if width < 1 or height < 1:
            raise ValueError(f"宽度和高度必须 >= 1，当前值: width={width}, height={height}")
        if stripe_width < 1:
            raise ValueError(f"stripe_width 必须 >= 1，当前值: {stripe_width}")
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation 必须为 'horizontal' 或 'vertical'，当前值: {orientation}"
            )

        progress = ProgressCallback(progress_callback)
        progress(0.5)

        # 生成条纹图案
        if orientation == "horizontal":
            indices = np.arange(height) // stripe_width
            stripes = (indices % 2)[:, np.newaxis] * np.ones((1, width))
        else:  # vertical
            indices = np.arange(width) // stripe_width
            stripes = np.ones((height, 1)) * (indices % 2)[np.newaxis, :]

        progress(1.0)
        self.update_metadata(
            width=width, height=height, stripe_width=stripe_width, orientation=orientation
        )
        return stripes.astype(np.int32)

    def save_png(self, pattern: NDArray[Any], output_path: Union[str, Path]) -> None:
        """
        保存条纹图案为 PNG 图像

        Args:
            pattern: 条纹图案矩阵
            output_path: 输出文件路径

        Raises:
            IOError: 文件写入失败
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为 0-255
        image_data = (pattern * 255).astype(np.uint8)
        image = Image.fromarray(image_data, mode="L")
        image.save(output_path)
