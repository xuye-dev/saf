"""
性能计时器模块

Author: 徐野
Date: 2025-11-23
"""

import time
from typing import Optional


class Timer:
    """性能计时器（支持上下文管理器）"""

    def __init__(self, name: str = "Timer") -> None:
        """
        初始化计时器

        Args:
            name: 计时器名称

        Example:
            >>> timer = Timer("数据压缩")
            >>> timer.start()
            >>> # ... 执行任务 ...
            >>> timer.stop()
            >>> print(timer.elapsed())
        """
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> "Timer":
        """
        启动计时器

        Returns:
            Timer: 返回自身，支持链式调用

        Example:
            >>> timer = Timer().start()
        """
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> float:
        """
        停止计时器并返回耗时

        Returns:
            float: 耗时（秒）

        Raises:
            RuntimeError: 计时器未启动

        Example:
            >>> timer = Timer().start()
            >>> time.sleep(1)
            >>> elapsed = timer.stop()
            >>> print(f"耗时: {elapsed:.2f} 秒")
        """
        if self._start_time is None:
            raise RuntimeError("计时器未启动，请先调用 start() 方法")

        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        return self._elapsed

    def elapsed(self) -> float:
        """
        获取当前已经过的时间（秒）

        如果计时器未停止，返回从启动到现在的时间
        如果计时器已停止，返回总耗时

        Returns:
            float: 耗时（秒）

        Raises:
            RuntimeError: 计时器未启动

        Example:
            >>> timer = Timer().start()
            >>> time.sleep(0.5)
            >>> print(timer.elapsed())  # 约 0.5 秒
        """
        if self._start_time is None:
            raise RuntimeError("计时器未启动")

        if self._end_time is None:
            # 计时器未停止，返回当前时间
            return time.perf_counter() - self._start_time
        else:
            # 计时器已停止，返回总耗时
            return self._elapsed

    def reset(self) -> None:
        """
        重置计时器

        Example:
            >>> timer = Timer()
            >>> timer.start()
            >>> timer.stop()
            >>> timer.reset()
        """
        self._start_time = None
        self._end_time = None
        self._elapsed = 0.0

    def __enter__(self) -> "Timer":
        """
        上下文管理器：进入时启动计时器

        Returns:
            Timer: 返回自身

        Example:
            >>> with Timer("数据压缩") as timer:
            ...     # 执行任务
            ...     pass
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """
        上下文管理器：退出时停止计时器

        Example:
            >>> with Timer("数据压缩") as timer:
            ...     pass
            >>> print(timer.elapsed())
        """
        self.stop()

    def __str__(self) -> str:
        """
        返回格式化的计时器信息

        Returns:
            str: 格式化字符串

        Example:
            >>> timer = Timer("测试")
            >>> timer.start()
            >>> time.sleep(0.1)
            >>> timer.stop()
            >>> print(timer)
            测试: 0.10 秒
        """
        try:
            elapsed = self.elapsed()
            return f"{self.name}: {format_time(elapsed)}"
        except RuntimeError:
            return f"{self.name}: 未启动"

    def __repr__(self) -> str:
        """
        返回计时器的字符串表示

        Returns:
            str: 字符串表示

        Example:
            >>> timer = Timer("测试")
            >>> repr(timer)
        """
        return f"Timer(name='{self.name}', elapsed={self._elapsed:.4f}s)"


def format_time(seconds: float) -> str:
    """
    格式化时间（自动选择单位：秒/毫秒/微秒）

    Args:
        seconds: 时间（秒）

    Returns:
        str: 格式化后的时间字符串

    Example:
        >>> print(format_time(0.001))
        1.00 毫秒
        >>> print(format_time(1.5))
        1.50 秒
    """
    if seconds < 1e-3:
        # 微秒
        return f"{seconds * 1e6:.2f} 微秒"
    elif seconds < 1:
        # 毫秒
        return f"{seconds * 1e3:.2f} 毫秒"
    elif seconds < 60:
        # 秒
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        # 分钟
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} 分 {secs:.2f} 秒"
    else:
        # 小时
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} 小时 {minutes} 分 {secs:.2f} 秒"
