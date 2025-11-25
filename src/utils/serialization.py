"""
数据序列化工具模块（支持 msgpack 和 JSON）

Author: 徐野
Date: 2025-11-23
"""

import json
from pathlib import Path
from typing import Any, Union, cast

import msgpack

from .file_io import ensure_parent_dir


def serialize_msgpack(data: Any) -> bytes:
    """
    使用 msgpack 序列化数据为字节

    Args:
        data: 要序列化的数据（支持 dict, list, int, float, str, bytes 等）

    Returns:
        bytes: 序列化后的字节数据

    Raises:
        TypeError: 数据类型不支持序列化

    Example:
        >>> data = {"name": "test", "value": 123}
        >>> binary_data = serialize_msgpack(data)
    """
    return cast(bytes, msgpack.packb(data, use_bin_type=True))


def deserialize_msgpack(data: bytes) -> Any:
    """
    使用 msgpack 反序列化字节数据

    Args:
        data: 序列化后的字节数据

    Returns:
        Any: 反序列化后的数据

    Raises:
        ValueError: 数据格式错误

    Example:
        >>> binary_data = b"\\x82\\xa4name\\xa4test\\xa5value{"
        >>> data = deserialize_msgpack(binary_data)
    """
    return msgpack.unpackb(data, raw=False)


def save_msgpack(file_path: Union[str, Path], data: Any) -> None:
    """
    将数据序列化为 msgpack 格式并保存到文件

    Args:
        file_path: 文件路径
        data: 要保存的数据

    Raises:
        IOError: 写入失败

    Example:
        >>> data = {"compression_ratio": 100.5, "algorithm": "symbolic"}
        >>> save_msgpack("result.msgpack", data)
    """
    file_path = Path(file_path)
    ensure_parent_dir(file_path)

    binary_data = serialize_msgpack(data)
    with open(file_path, "wb") as f:
        f.write(binary_data)


def load_msgpack(file_path: Union[str, Path]) -> Any:
    """
    从文件加载 msgpack 格式的数据

    Args:
        file_path: 文件路径

    Returns:
        Any: 反序列化后的数据

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据格式错误

    Example:
        >>> data = load_msgpack("result.msgpack")
        >>> print(data["compression_ratio"])
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "rb") as f:
        binary_data = f.read()

    return deserialize_msgpack(binary_data)


def serialize_json(data: Any, indent: int = 2) -> str:
    """
    使用 JSON 序列化数据为字符串

    Args:
        data: 要序列化的数据
        indent: 缩进空格数，默认 2（美化输出）

    Returns:
        str: JSON 字符串

    Raises:
        TypeError: 数据类型不支持序列化

    Example:
        >>> data = {"name": "test", "value": 123}
        >>> json_str = serialize_json(data)
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def deserialize_json(json_str: str) -> Any:
    """
    使用 JSON 反序列化字符串

    Args:
        json_str: JSON 字符串

    Returns:
        Any: 反序列化后的数据

    Raises:
        ValueError: JSON 格式错误

    Example:
        >>> json_str = '{"name": "test", "value": 123}'
        >>> data = deserialize_json(json_str)
    """
    return json.loads(json_str)


def save_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """
    将数据序列化为 JSON 格式并保存到文件

    Args:
        file_path: 文件路径
        data: 要保存的数据
        indent: 缩进空格数，默认 2

    Raises:
        IOError: 写入失败

    Example:
        >>> data = {"compression_ratio": 100.5, "algorithm": "symbolic"}
        >>> save_json("result.json", data)
    """
    file_path = Path(file_path)
    ensure_parent_dir(file_path)

    json_str = serialize_json(data, indent=indent)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    从文件加载 JSON 格式的数据

    Args:
        file_path: 文件路径

    Returns:
        Any: 反序列化后的数据

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式错误

    Example:
        >>> data = load_json("result.json")
        >>> print(data["compression_ratio"])
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        json_str = f.read()

    return deserialize_json(json_str)
