"""
数学序列生成器（斐波那契、素数、π位数、自定义公式）

Author: 徐野
Date: 2025-11-23
"""

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sympy import N
from sympy import pi as sympy_pi

from .base import BaseGenerator, ProgressCallback


class FibonacciGenerator(BaseGenerator):
    """斐波那契数列生成器"""

    def __init__(self) -> None:
        """初始化斐波那契生成器"""
        super().__init__(name="Fibonacci")
        self.update_metadata(data_type="sequence", pattern_type="recursive")

    def generate(
        self,
        n: int,
        use_bigint: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成斐波那契数列的前 n 项

        Args:
            n: 生成的项数（必须 >= 1）
            use_bigint: 是否使用大整数（避免溢出），默认 False
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 斐波那契数列（1D 数组）

        Raises:
            ValueError: n < 1

        Example:
            >>> gen = FibonacciGenerator()
            >>> data = gen.generate(n=10)
            >>> print(data)
            [0 1 1 2 3 5 8 13 21 34]
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1，当前值: {n}")

        progress = ProgressCallback(progress_callback)
        dtype = object if use_bigint else np.int64

        # 初始化数组
        fib = np.zeros(n, dtype=dtype)
        if n >= 1:
            fib[0] = 0
        if n >= 2:
            fib[1] = 1

        # 计算斐波那契数列
        for i in range(2, n):
            fib[i] = fib[i - 1] + fib[i - 2]
            if i % 100 == 0:  # 每 100 步更新一次进度
                progress(i / n)

        progress(1.0)
        self.update_metadata(n=n, use_bigint=use_bigint)
        return fib


class PrimeGenerator(BaseGenerator):
    """素数序列生成器（使用埃拉托斯特尼筛法）"""

    def __init__(self) -> None:
        """初始化素数生成器"""
        super().__init__(name="Prime")
        self.update_metadata(data_type="sequence", pattern_type="prime")

    def generate(
        self,
        n: Optional[int] = None,
        max_value: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成素数序列

        Args:
            n: 生成前 n 个素数（与 max_value 二选一）
            max_value: 生成不超过 max_value 的所有素数（与 n 二选一）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 素数序列（1D 数组）

        Raises:
            ValueError: n 和 max_value 都未指定，或同时指定

        Example:
            >>> gen = PrimeGenerator()
            >>> primes = gen.generate(n=10)
            >>> print(primes)
            [2 3 5 7 11 13 17 19 23 29]
        """
        if (n is None and max_value is None) or (n is not None and max_value is not None):
            raise ValueError("必须指定 n 或 max_value 之一（且只能指定一个）")

        progress = ProgressCallback(progress_callback)

        if max_value is not None:
            # 使用埃拉托斯特尼筛法生成不超过 max_value 的所有素数
            if max_value < 2:
                return np.array([], dtype=np.int64)

            is_prime = np.ones(max_value + 1, dtype=bool)
            is_prime[0] = is_prime[1] = False

            for i in range(2, int(np.sqrt(max_value)) + 1):
                if is_prime[i]:
                    is_prime[i * i : max_value + 1 : i] = False
                progress(i / int(np.sqrt(max_value)))

            primes_array = np.where(is_prime)[0]
            progress(1.0)
            self.update_metadata(max_value=max_value, count=len(primes_array))
            return primes_array

        # 生成前 n 个素数（使用试除法）
        assert n is not None
        if n < 1:
            raise ValueError(f"n 必须 >= 1，当前值: {n}")

        primes_list: list[int] = []
        candidate = 2

        while len(primes_list) < n:
            is_prime_flag = True
            for p in primes_list:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime_flag = False
                    break

            if is_prime_flag:
                primes_list.append(candidate)

            candidate += 1 if candidate == 2 else 2  # 跳过偶数

            if len(primes_list) % 100 == 0:
                progress(len(primes_list) / n)

        progress(1.0)
        self.update_metadata(n=n)
        return np.array(primes_list, dtype=np.int64)


class PiDigitsGenerator(BaseGenerator):
    """π 的十进制位数生成器（使用 SymPy）"""

    def __init__(self) -> None:
        """初始化 π 位数生成器"""
        super().__init__(name="PiDigits")
        self.update_metadata(data_type="sequence", pattern_type="transcendental")

    def generate(
        self,
        n: int,
        include_decimal_point: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成 π 的前 n 位十进制数字

        Args:
            n: 生成的位数（必须 >= 1）
            include_decimal_point: 是否包含小数点前的 '3'，默认 False（仅小数部分）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: π 的位数序列（1D 数组，每个元素为 0-9 的整数）

        Raises:
            ValueError: n < 1
            TimeoutError: 计算超时（n 过大）

        Example:
            >>> gen = PiDigitsGenerator()
            >>> digits = gen.generate(n=10, include_decimal_point=True)
            >>> print(digits)
            [3 1 4 1 5 9 2 6 5 3]
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1，当前值: {n}")

        progress = ProgressCallback(progress_callback)
        progress(0.1)

        # 使用 SymPy 计算 π 的高精度值
        # 需要额外的精度以确保最后一位正确
        precision = n + 10
        pi_str = str(N(sympy_pi, precision))

        progress(0.8)

        # 移除小数点，提取数字
        pi_digits_str = pi_str.replace(".", "")

        # 转换为整数数组
        if include_decimal_point:
            digits = [int(d) for d in pi_digits_str[:n]]
        else:
            # 跳过整数部分的 '3'
            digits = [int(d) for d in pi_digits_str[1 : n + 1]]

        progress(1.0)
        self.update_metadata(n=n, include_decimal_point=include_decimal_point)
        return np.array(digits, dtype=np.int32)


class FormulaGenerator(BaseGenerator):
    """自定义数学公式序列生成器"""

    def __init__(self, formula: Callable[[int], float], formula_name: str = "Custom") -> None:
        """
        初始化公式生成器

        Args:
            formula: 公式函数，接收索引 i (0-based)，返回该位置的值
            formula_name: 公式名称（用于日志和元数据）

        Example:
            >>> gen = FormulaGenerator(lambda i: i**2, formula_name="Squares")
            >>> data = gen.generate(n=10)
            >>> print(data)
            [0 1 4 9 16 25 36 49 64 81]
        """
        super().__init__(name=f"Formula-{formula_name}")
        self.formula = formula
        self.formula_name = formula_name
        self.update_metadata(
            data_type="sequence", pattern_type="formula", formula_name=formula_name
        )

    def generate(
        self,
        n: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> NDArray[Any]:
        """
        生成公式序列的前 n 项

        Args:
            n: 生成的项数（必须 >= 1）
            progress_callback: 进度回调函数（可选）

        Returns:
            np.ndarray: 公式序列（1D 数组）

        Raises:
            ValueError: n < 1

        Example:
            >>> gen = FormulaGenerator(lambda i: 2 * i + 1, formula_name="Odd")
            >>> data = gen.generate(n=5)
            >>> print(data)
            [1 3 5 7 9]
        """
        if n < 1:
            raise ValueError(f"n 必须 >= 1，当前值: {n}")

        progress = ProgressCallback(progress_callback)

        # 计算序列
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            result[i] = self.formula(i)
            if i % 100 == 0:
                progress(i / n)

        progress(1.0)
        self.update_metadata(n=n)
        return result
