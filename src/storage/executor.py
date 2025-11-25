"""
算法执行器（用于重建数据）

Author: 徐野
Date: 2025-11-23
"""

import gzip
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from ..generators.fractal import JuliaGenerator, MandelbrotGenerator
from ..generators.sequence import FibonacciGenerator, PiDigitsGenerator, PrimeGenerator
from ..utils.logger import get_logger
from .format import StorageFormat

logger = get_logger(__name__)


class AlgorithmExecutor:
    """
    算法执行器

    根据算法描述符重建原始数据，支持：
    - 序列算法：fibonacci, arithmetic, geometric, polynomial, primes, pi_digits
    - 分形算法：mandelbrot, julia
    - 降级压缩：gzip
    """

    def __init__(self) -> None:
        """初始化算法执行器"""
        # 初始化生成器实例
        self.fibonacci_gen = FibonacciGenerator()
        self.prime_gen = PrimeGenerator()
        self.pi_gen = PiDigitsGenerator()
        self.mandelbrot_gen = MandelbrotGenerator()
        self.julia_gen = JuliaGenerator()

        # 注册算法执行函数
        self._executors = {
            "fibonacci": self._execute_fibonacci,
            "recursive": self._execute_recursive,  # 通用递归规律
            "arithmetic": self._execute_arithmetic,
            "geometric": self._execute_geometric,
            "polynomial": self._execute_polynomial,
            "primes": self._execute_primes,
            "pi_digits": self._execute_pi_digits,
            "mandelbrot": self._execute_mandelbrot,
            "julia": self._execute_julia,
            "checkerboard": self._execute_checkerboard,  # 棋盘图案
            "stripes": self._execute_stripes,  # 条纹图案
            "perlin_noise": self._execute_perlin_noise,  # Perlin 噪声
            "gzip": self._execute_gzip,
        }

    def rebuild_data(self, storage_format: StorageFormat) -> NDArray[Any]:
        """
        根据存储格式重建数据

        Args:
            storage_format: 存储格式实例

        Returns:
            NDArray[Any]: 重建的数据（NumPy 数组）

        Raises:
            ValueError: 不支持的算法类型
            RuntimeError: 数据重建失败

        Example:
            >>> executor = AlgorithmExecutor()
            >>> storage = StorageFormat.load("data.saf")
            >>> data = executor.rebuild_data(storage)
        """
        algorithm_type = storage_format.algorithm.type
        parameters = storage_format.algorithm.parameters

        logger.info(f"开始重建数据，算法类型: {algorithm_type}")

        # 查找对应的执行函数
        if algorithm_type not in self._executors:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")

        executor_func = self._executors[algorithm_type]

        try:
            # 执行算法重建数据
            data = executor_func(parameters)

            # 验证数据形状
            expected_shape = tuple(storage_format.metadata.shape)
            if data.shape != expected_shape:
                raise RuntimeError(
                    f"重建数据形状不匹配，期望: {expected_shape}, 实际: {data.shape}"
                )

            # 确保 dtype 与原始数据一致（用于哈希验证）
            expected_dtype = storage_format.metadata.dtype
            if str(data.dtype) != expected_dtype:
                logger.debug(f"转换 dtype: {data.dtype} → {expected_dtype}")
                data = data.astype(expected_dtype)

            logger.info(f"数据重建成功，形状: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"数据重建失败: {e}")
            raise RuntimeError(f"数据重建失败: {e}") from e

    # ========== 序列算法执行函数 ==========

    def _execute_fibonacci(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行斐波那契数列生成"""
        n = params["n"]
        use_bigint = params.get("use_bigint", False)
        logger.debug(f"生成斐波那契数列，n={n}, use_bigint={use_bigint}")
        return self.fibonacci_gen.generate(n=n, use_bigint=use_bigint)

    def _execute_recursive(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行通用递归规律序列生成（a[n] = k * a[n-1] + b * a[n-2]）"""
        first_term = params["first_term"]
        second_term = params["second_term"]
        k = params["coefficient_k"]
        b = params["coefficient_b"]
        n = params["n"]

        logger.debug(f"生成递归序列，n={n}, k={k}, b={b}")

        # 检查系数是否为整数（避免大数值浮点精度问题）
        k_is_int = abs(k - round(k)) < 1e-9
        b_is_int = abs(b - round(b)) < 1e-9

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(first_term)
        if n > 1:
            result[1] = int(second_term)

        if k_is_int and b_is_int:
            # 使用整数运算（避免大数值浮点精度丢失）
            k_int = int(round(k))
            b_int = int(round(b))
            for i in range(2, n):
                result[i] = k_int * result[i - 1] + b_int * result[i - 2]
        else:
            # 使用浮点运算
            for i in range(2, n):
                result[i] = int(k * result[i - 1] + b * result[i - 2])

        return result

    def _execute_arithmetic(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行等差数列生成"""
        n = params["n"]
        # 支持两种参数名：检测器使用 first_term/common_difference，旧版本使用 start/diff
        start = params.get("first_term", params.get("start"))
        diff = params.get("common_difference", params.get("diff"))
        logger.debug(f"生成等差数列，n={n}, start={start}, diff={diff}")
        return np.array([start + i * diff for i in range(n)], dtype=np.int64)

    def _execute_geometric(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行等比数列生成"""
        n = params["n"]
        # 支持两种参数名：检测器使用 first_term/common_ratio，旧版本使用 start/ratio
        start = params.get("first_term", params.get("start"))
        ratio = params.get("common_ratio", params.get("ratio"))
        logger.debug(f"生成等比数列，n={n}, start={start}, ratio={ratio}")
        return np.array([start * (ratio**i) for i in range(n)])

    def _execute_polynomial(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行多项式序列生成"""
        n = params["n"]
        coefficients = params["coefficients"]
        logger.debug(f"生成多项式序列，n={n}, coefficients={coefficients}")

        # 使用多项式公式: f(x) = c0 + c1*x + c2*x^2 + ...
        x = np.arange(n)
        result = np.zeros(n, dtype=np.float64)
        for i, coef in enumerate(coefficients):
            result += coef * (x**i)

        # 使用四舍五入而非截断，避免浮点误差导致的偏差
        return np.round(result)

    def _execute_primes(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行素数序列生成"""
        n = params.get("n")
        max_value = params.get("max_value")
        logger.debug(f"生成素数序列，n={n}, max_value={max_value}")
        return self.prime_gen.generate(n=n, max_value=max_value)

    def _execute_pi_digits(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行 π 位数生成"""
        n = params["n"]
        logger.debug(f"生成 π 位数，n={n}")
        return self.pi_gen.generate(n=n)

    # ========== 分形算法执行函数 ==========

    def _execute_mandelbrot(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行 Mandelbrot 集合生成"""
        width = params["width"]
        height = params["height"]
        max_iter = params["max_iter"]
        x_min = params["x_min"]
        x_max = params["x_max"]
        y_min = params["y_min"]
        y_max = params["y_max"]

        logger.debug(
            f"生成 Mandelbrot 集合，分辨率={width}x{height}, max_iter={max_iter}"
        )

        return self.mandelbrot_gen.generate(
            width=width,
            height=height,
            max_iter=max_iter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    def _execute_julia(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行 Julia 集合生成"""
        width = params["width"]
        height = params["height"]
        max_iter = params["max_iter"]
        c_real = params["c_real"]
        c_imag = params["c_imag"]
        x_min = params["x_min"]
        x_max = params["x_max"]
        y_min = params["y_min"]
        y_max = params["y_max"]

        logger.debug(
            f"生成 Julia 集合，分辨率={width}x{height}, c={c_real}+{c_imag}j"
        )

        # JuliaGenerator 的 c 参数需要在构造函数中指定
        julia_gen = JuliaGenerator(c=complex(c_real, c_imag))
        return julia_gen.generate(
            width=width,
            height=height,
            max_iter=max_iter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    # ========== 2D 图案执行函数 ==========

    def _execute_checkerboard(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行棋盘图案重建"""
        height = params["height"]
        width = params["width"]
        cell_size = params["cell_size"]
        color_0 = params["color_0"]
        color_1 = params["color_1"]

        logger.debug(
            f"重建棋盘图案，尺寸={height}x{width}, 格子大小={cell_size}"
        )

        # 生成棋盘图案
        data = np.zeros((height, width), dtype=np.int32)
        for i in range(height):
            for j in range(width):
                if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                    data[i, j] = color_0
                else:
                    data[i, j] = color_1

        return data

    def _execute_stripes(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行条纹图案重建"""
        height = params["height"]
        width = params["width"]
        direction = params["direction"]
        stripe_width = params["stripe_width"]
        color_0 = params["color_0"]
        color_1 = params["color_1"]

        logger.debug(
            f"重建条纹图案，尺寸={height}x{width}, 方向={direction}, 条纹宽度={stripe_width}"
        )

        # 生成条纹图案
        data = np.zeros((height, width), dtype=np.int32)

        if direction == "horizontal":
            for i in range(height):
                if (i // stripe_width) % 2 == 0:
                    data[i, :] = color_0
                else:
                    data[i, :] = color_1
        else:  # vertical
            for j in range(width):
                if (j // stripe_width) % 2 == 0:
                    data[:, j] = color_0
                else:
                    data[:, j] = color_1

        return data

    def _execute_perlin_noise(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行 Perlin 噪声重建"""
        width = params["width"]
        height = params["height"]
        scale = params["scale"]
        octaves = params["octaves"]
        persistence = params["persistence"]
        lacunarity = params["lacunarity"]
        seed = params["seed"]

        logger.debug(
            f"重建 Perlin 噪声，尺寸={height}x{width}, scale={scale}, "
            f"octaves={octaves}, seed={seed}"
        )

        try:
            import noise as noise_lib  # type: ignore
        except ImportError as e:
            raise RuntimeError("需要安装 noise 库: pip install noise") from e

        # 生成 Perlin 噪声
        data = np.zeros((height, width), dtype=np.float64)

        for y in range(height):
            for x in range(width):
                data[y, x] = noise_lib.pnoise2(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=width,
                    repeaty=height,
                    base=seed,
                )

        return data

    # ========== 降级压缩执行函数 ==========

    def _execute_gzip(self, params: Dict[str, Any]) -> NDArray[Any]:
        """执行 gzip 解压"""
        compressed_data = bytes(params["compressed_data"])
        dtype = params["dtype"]
        shape = tuple(params["shape"])

        logger.debug(f"gzip 解压，原始大小={len(compressed_data)} 字节")

        # 解压数据
        decompressed_data = gzip.decompress(compressed_data)

        # 转换为 NumPy 数组
        data = np.frombuffer(decompressed_data, dtype=dtype).reshape(shape)

        return data


def rebuild_data_from_file(file_path: str) -> NDArray[Any]:
    """
    从 .saf 文件重建数据（便捷函数）

    Args:
        file_path: .saf 文件路径

    Returns:
        NDArray[Any]: 重建的数据

    Example:
        >>> data = rebuild_data_from_file("compressed.saf")
        >>> print(data.shape)
    """
    storage = StorageFormat.load(file_path)
    executor = AlgorithmExecutor()
    return executor.rebuild_data(storage)
