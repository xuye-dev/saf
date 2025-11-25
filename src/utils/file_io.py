"""
文件 I/O 工具模块

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Union


def read_binary(file_path: Union[str, Path]) -> bytes:
    """
    读取二进制文件

    Args:
        file_path: 文件路径

    Returns:
        bytes: 文件内容（字节）

    Raises:
        FileNotFoundError: 文件不存在
        IOError: 读取失败

    Example:
        >>> data = read_binary("data.bin")
        >>> print(len(data))
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "rb") as f:
        return f.read()


def read_text(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    读取文本文件

    Args:
        file_path: 文件路径
        encoding: 文件编码，默认 UTF-8

    Returns:
        str: 文件内容（文本）

    Raises:
        FileNotFoundError: 文件不存在
        IOError: 读取失败

    Example:
        >>> text = read_text("readme.txt")
        >>> print(text[:100])
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_binary(file_path: Union[str, Path], data: bytes) -> None:
    """
    写入二进制文件

    Args:
        file_path: 文件路径
        data: 要写入的字节数据

    Raises:
        IOError: 写入失败

    Example:
        >>> data = b"\\x00\\x01\\x02\\x03"
        >>> write_binary("output.bin", data)
    """
    file_path = Path(file_path)
    ensure_parent_dir(file_path)

    with open(file_path, "wb") as f:
        f.write(data)


def write_text(file_path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    """
    写入文本文件

    Args:
        file_path: 文件路径
        text: 要写入的文本
        encoding: 文件编码，默认 UTF-8

    Raises:
        IOError: 写入失败

    Example:
        >>> write_text("output.txt", "Hello World")
    """
    file_path = Path(file_path)
    ensure_parent_dir(file_path)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(text)


def ensure_parent_dir(file_path: Union[str, Path]) -> None:
    """
    确保文件的父目录存在（如不存在则创建）

    Args:
        file_path: 文件路径

    Example:
        >>> ensure_parent_dir("path/to/file.txt")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(dir_path: Union[str, Path]) -> None:
    """
    确保目录存在（如不存在则创建）

    Args:
        dir_path: 目录路径

    Example:
        >>> ensure_dir("logs")
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）

    Args:
        file_path: 文件路径

    Returns:
        int: 文件大小（字节）

    Raises:
        FileNotFoundError: 文件不存在

    Example:
        >>> size = get_file_size("data.bin")
        >>> print(f"文件大小: {size} 字节")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小（自动选择单位：B/KB/MB/GB）

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        str: 格式化后的文件大小字符串

    Example:
        >>> print(format_file_size(1024))
        1.00 KB
        >>> print(format_file_size(1048576))
        1.00 MB
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def get_relative_path(file_path: Union[str, Path], base_path: Union[str, Path]) -> Path:
    """
    获取相对路径

    Args:
        file_path: 文件路径
        base_path: 基准路径

    Returns:
        Path: 相对路径

    Example:
        >>> rel_path = get_relative_path("/a/b/c/file.txt", "/a/b")
        >>> print(rel_path)
        c/file.txt
    """
    file_path = Path(file_path).resolve()
    base_path = Path(base_path).resolve()
    return file_path.relative_to(base_path)
