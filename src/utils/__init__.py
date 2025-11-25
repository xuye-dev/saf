"""
Utils module - Configuration, logging, file I/O, serialization, timer, and hash utilities

Author: 徐野
Date: 2025-11-23
"""

# Configuration management
from .config import (
    CompressionConfig,
    Config,
    ConfigLoader,
    DetectionConfig,
    LoggingConfig,
    PerformanceConfig,
    get_config,
    load_config,
)

# File I/O utilities
from .file_io import (
    ensure_dir,
    ensure_parent_dir,
    format_file_size,
    get_file_size,
    get_relative_path,
    read_binary,
    read_text,
    write_binary,
    write_text,
)

# Hash utilities
from .hash_utils import (
    compute_file_hash,
    compute_hash,
    hash_comparison,
    verify_file_hash,
    verify_hash,
)

# Logging system
from .logger import (
    get_default_logger,
    get_logger,
    init_default_logger,
    setup_logger,
)

# Serialization utilities
from .serialization import (
    deserialize_json,
    deserialize_msgpack,
    load_json,
    load_msgpack,
    save_json,
    save_msgpack,
    serialize_json,
    serialize_msgpack,
)

# Performance timer
from .timer import Timer, format_time

__all__ = [
    # Configuration management
    "Config",
    "LoggingConfig",
    "PerformanceConfig",
    "CompressionConfig",
    "DetectionConfig",
    "ConfigLoader",
    "load_config",
    "get_config",
    # Logging system
    "setup_logger",
    "get_logger",
    "init_default_logger",
    "get_default_logger",
    # File I/O utilities
    "read_binary",
    "read_text",
    "write_binary",
    "write_text",
    "ensure_parent_dir",
    "ensure_dir",
    "get_file_size",
    "format_file_size",
    "get_relative_path",
    # Serialization utilities
    "serialize_msgpack",
    "deserialize_msgpack",
    "save_msgpack",
    "load_msgpack",
    "serialize_json",
    "deserialize_json",
    "save_json",
    "load_json",
    # Performance timer
    "Timer",
    "format_time",
    # Hash utilities
    "compute_hash",
    "compute_file_hash",
    "verify_hash",
    "verify_file_hash",
    "hash_comparison",
]
