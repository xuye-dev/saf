"""
哈希计算工具模块（用于数据验证）

Author: 徐野
Date: 2025-11-23
"""

import hashlib
from pathlib import Path
from typing import Union


def compute_hash(data: bytes) -> str:
    """
    计算字节数据的 SHA-256 哈希值

    Args:
        data: 字节数据

    Returns:
        str: SHA-256 哈希值（64 个十六进制字符）

    Example:
        >>> data = b"Hello, World!"
        >>> hash_value = compute_hash(data)
        >>> print(hash_value)
        dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f
    """
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    计算文件的 SHA-256 哈希值（支持大文件分块读取）

    Args:
        file_path: 文件路径
        chunk_size: 分块读取大小（字节），默认 8KB

    Returns:
        str: SHA-256 哈希值（64 个十六进制字符）

    Raises:
        FileNotFoundError: 文件不存在

    Example:
        >>> hash_value = compute_file_hash("data.bin")
        >>> print(hash_value)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_hash(data: bytes, expected_hash: str) -> bool:
    """
    验证字节数据的哈希值是否匹配

    Args:
        data: 字节数据
        expected_hash: 期望的哈希值（64 个十六进制字符）

    Returns:
        bool: 哈希值是否匹配

    Example:
        >>> data = b"Hello, World!"
        >>> expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        >>> print(verify_hash(data, expected))
        True
    """
    actual_hash = compute_hash(data)
    return actual_hash.lower() == expected_hash.lower()


def verify_file_hash(
    file_path: Union[str, Path], expected_hash: str, chunk_size: int = 8192
) -> bool:
    """
    验证文件的哈希值是否匹配

    Args:
        file_path: 文件路径
        expected_hash: 期望的哈希值（64 个十六进制字符）
        chunk_size: 分块读取大小（字节），默认 8KB

    Returns:
        bool: 哈希值是否匹配

    Raises:
        FileNotFoundError: 文件不存在

    Example:
        >>> expected = "abc123..."
        >>> print(verify_file_hash("data.bin", expected))
        True
    """
    actual_hash = compute_file_hash(file_path, chunk_size)
    return actual_hash.lower() == expected_hash.lower()


def hash_comparison(hash1: str, hash2: str) -> bool:
    """
    比较两个哈希值是否相等（不区分大小写）

    Args:
        hash1: 第一个哈希值
        hash2: 第二个哈希值

    Returns:
        bool: 哈希值是否相等

    Example:
        >>> hash1 = "ABCDEF1234567890"
        >>> hash2 = "abcdef1234567890"
        >>> print(hash_comparison(hash1, hash2))
        True
    """
    return hash1.lower() == hash2.lower()
