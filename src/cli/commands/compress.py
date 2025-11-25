"""
压缩命令实现

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path

import click

from ...storage.compressor import compress_file
from ...utils.logger import get_logger
from ..ui import echo_error, echo_info, print_compression_result

logger = get_logger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="输出文件路径（默认为输入文件名 + .saf 后缀）",
)
@click.option(
    "-t",
    "--threshold",
    "confidence_threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="置信度阈值（低于此值将回退到 gzip）",
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
def compress(
    input_file: str,
    output_file: str | None,
    confidence_threshold: float,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    压缩数据文件

    将 .npy 格式的数据文件压缩为 .saf 格式的算法式存储文件。

    \b
    示例:
      saf compress data.npy                    # 输出 data.saf
      saf compress data.npy -o output.saf      # 指定输出文件
      saf compress data.npy -t 0.9             # 设置置信度阈值
    """
    input_path = Path(input_file)

    # 检查输入文件格式
    if input_path.suffix.lower() != ".npy":
        echo_error(f"输入文件必须是 .npy 格式: {input_file}")
        raise SystemExit(1)

    # 确定输出文件路径
    if output_file is None:
        output_path = input_path.with_suffix(".saf")
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() != ".saf":
            output_path = output_path.with_suffix(".saf")

    # 检查输出文件是否已存在
    if output_path.exists():
        if not click.confirm(f"输出文件已存在: {output_path}，是否覆盖？"):
            echo_info("操作已取消")
            raise SystemExit(0)

    if not quiet:
        echo_info(f"正在压缩: {input_path}")

    try:
        # 执行压缩
        result = compress_file(
            input_path=input_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
        )

        if not quiet:
            print_compression_result(
                input_file=str(input_path),
                output_file=str(output_path),
                original_size=result.original_size,
                compressed_size=result.compressed_size,
                compression_ratio=result.compression_ratio,
                algorithm_type=result.algorithm_type,
                compression_time=result.compression_time,
            )

        if verbose:
            echo_info(f"置信度: {result.metadata.get('confidence', 'N/A')}")
            echo_info(f"方法: {result.metadata.get('method', 'N/A')}")

    except FileNotFoundError as e:
        echo_error(f"文件不存在: {e}")
        raise SystemExit(1)
    except Exception as e:
        echo_error(f"压缩失败: {e}")
        logger.exception("压缩过程中发生错误")
        raise SystemExit(1)
