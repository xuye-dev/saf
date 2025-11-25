"""
解压命令实现

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path

import click

from ...storage.decompressor import decompress_file
from ...utils.logger import get_logger
from ...utils.timer import Timer
from ..ui import echo_error, echo_info, print_decompression_result

logger = get_logger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="输出文件路径（默认为输入文件名 + .npy 后缀）",
)
@click.option(
    "--no-verify",
    is_flag=True,
    default=False,
    help="跳过哈希验证",
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
    help="静默模式（仅输出错误）",
)
def decompress(
    input_file: str,
    output_file: str | None,
    no_verify: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    解压数据文件

    将 .saf 格式的算法式存储文件解压为 .npy 格式的数据文件。

    \b
    示例:
      saf decompress data.saf                  # 输出 data.npy
      saf decompress data.saf -o output.npy    # 指定输出文件
      saf decompress data.saf --no-verify      # 跳过哈希验证
    """
    input_path = Path(input_file)

    # 检查输入文件格式
    if input_path.suffix.lower() != ".saf":
        echo_error(f"输入文件必须是 .saf 格式: {input_file}")
        raise SystemExit(1)

    # 确定输出文件路径
    if output_file is None:
        output_path = input_path.with_suffix(".npy")
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() != ".npy":
            output_path = output_path.with_suffix(".npy")

    # 检查输出文件是否已存在
    if output_path.exists():
        if not click.confirm(f"输出文件已存在: {output_path}，是否覆盖？"):
            echo_info("操作已取消")
            raise SystemExit(0)

    if not quiet:
        echo_info(f"正在解压: {input_path}")

    timer = Timer()
    timer.start()

    try:
        # 执行解压
        verification = decompress_file(
            input_path=input_path,
            output_path=output_path,
            verify_hash=not no_verify,
        )

        decompression_time = timer.stop()

        if not quiet:
            print_decompression_result(
                input_file=str(input_path),
                output_file=str(output_path),
                is_valid=verification.is_valid,
                decompression_time=decompression_time,
            )

        if verbose:
            echo_info(f"原始哈希: {verification.original_hash[:32]}...")
            echo_info(f"重建哈希: {verification.reconstructed_hash[:32]}...")

        if not verification.is_valid:
            raise SystemExit(1)

    except FileNotFoundError as e:
        echo_error(f"文件不存在: {e}")
        raise SystemExit(1)
    except RuntimeError as e:
        echo_error(f"解压失败: {e}")
        raise SystemExit(1)
    except Exception as e:
        echo_error(f"解压失败: {e}")
        logger.exception("解压过程中发生错误")
        raise SystemExit(1)
