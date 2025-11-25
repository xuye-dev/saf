"""
验证命令实现

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Optional

import click

from ...utils.logger import get_logger
from ...verification.verifier import verify_file
from ..ui import (
    echo_error,
    echo_info,
    echo_success,
    print_verification_result,
)

logger = get_logger(__name__)


@click.command()
@click.argument("saf_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--original",
    "original_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="原始数据文件路径（用于对比验证）",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["hash", "byte_by_byte"]),
    default="hash",
    show_default=True,
    help="验证方法",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="显示详细信息",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="静默模式（仅返回退出码）",
)
def verify(
    saf_file: str,
    original_file: Optional[str],
    method: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    验证压缩文件的完整性

    检查 .saf 文件是否可以正确解压并还原原始数据。

    \b
    示例:
      saf verify data.saf                      # 使用存储的哈希验证
      saf verify data.saf --original data.npy  # 与原始文件对比
      saf verify data.saf -m byte_by_byte      # 逐字节对比验证
    """
    saf_path = Path(saf_file)

    # 检查输入文件格式
    if saf_path.suffix.lower() != ".saf":
        echo_error(f"输入文件必须是 .saf 格式: {saf_file}")
        raise SystemExit(1)

    if not quiet:
        echo_info(f"正在验证: {saf_path}")

    try:
        # 执行验证
        result = verify_file(
            saf_path=saf_path,
            original_path=original_file,
            method=method,
        )

        if not quiet:
            print_verification_result(
                file_path=str(saf_path),
                is_valid=result.is_valid,
                original_hash=result.original_hash,
                reconstructed_hash=result.reconstructed_hash,
                verification_time=result.verification_time,
                error_message=result.error_message,
            )

        if verbose and result.is_valid:
            echo_info(f"验证方法: {method}")
            if original_file:
                echo_info(f"对比文件: {original_file}")

        # 根据验证结果设置退出码
        if result.is_valid:
            if quiet:
                echo_success("验证通过")
            raise SystemExit(0)
        else:
            if quiet:
                echo_error("验证失败")
            raise SystemExit(1)

    except FileNotFoundError as e:
        echo_error(f"文件不存在: {e}")
        raise SystemExit(1)
    except Exception as e:
        echo_error(f"验证失败: {e}")
        logger.exception("验证过程中发生错误")
        raise SystemExit(1)
