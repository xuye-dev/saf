"""
日志系统模块（基于 Python logging）

Author: 徐野
Date: 2025-11-23
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


def setup_logger(
    name: str = "saf",
    config: Optional[LoggingConfig] = None,
) -> logging.Logger:
    """
    配置并返回日志记录器

    Args:
        name: 日志记录器名称
        config: 日志配置对象，如果为 None 则使用默认配置

    Returns:
        logging.Logger: 配置好的日志记录器

    Example:
        >>> logger = setup_logger()
        >>> logger.info("这是一条信息日志")
    """
    # 使用默认配置或提供的配置
    if config is None:
        config = LoggingConfig()

    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 定义日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 添加控制台处理器
    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器（带日志轮转）
    if config.file_path:
        # 确保日志目录存在
        log_file_path = Path(config.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建 RotatingFileHandler
        max_bytes = config.max_file_size_mb * 1024 * 1024  # 转换为字节
        file_handler = RotatingFileHandler(
            filename=config.file_path,
            maxBytes=max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到父记录器
    logger.propagate = False

    return logger


def get_logger(name: str = "saf") -> logging.Logger:
    """
    获取已配置的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 日志记录器

    Example:
        >>> logger = get_logger()
        >>> logger.debug("调试信息")
    """
    logger = logging.getLogger(name)

    # 如果还未配置，使用默认配置
    if not logger.handlers:
        return setup_logger(name)

    return logger


# 模块级别的默认日志记录器
_default_logger: Optional[logging.Logger] = None


def init_default_logger(config: Optional[LoggingConfig] = None) -> None:
    """
    初始化默认日志记录器

    Args:
        config: 日志配置对象

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> init_default_logger(config.logging)
    """
    global _default_logger
    _default_logger = setup_logger("saf", config)


def get_default_logger() -> logging.Logger:
    """
    获取默认日志记录器

    Returns:
        logging.Logger: 默认日志记录器

    Example:
        >>> logger = get_default_logger()
        >>> logger.info("默认日志记录器")
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger("saf")
    return _default_logger
