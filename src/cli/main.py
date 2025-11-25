"""
命令行工具主入口

提供 saf 命令行工具的主入口点，整合所有子命令。

Author: 徐野
Date: 2025-11-23
"""

import click

from .commands import benchmark, compress, decompress, verify

# 版本号
__version__ = "0.1.6"


@click.group()
@click.version_option(version=__version__, prog_name="saf")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="启用调试模式（显示详细日志）",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """
    算法式存储系统 (Algorithmic Storage System)

    一个基于 Kolmogorov 复杂度理论的超高压缩比存储系统，
    通过存储"生成数据的算法"而非数据本身来实现极致压缩。

    \b
    支持的数据类型:
      - 数学序列（斐波那契、素数、等差、等比、多项式）
      - 分形图像（Mandelbrot、Julia）

    \b
    快速开始:
      saf compress data.npy           # 压缩文件
      saf decompress data.saf         # 解压文件
      saf verify data.saf             # 验证文件
      saf benchmark data/             # 批量测试

    \b
    更多信息:
      saf <command> --help            # 查看子命令帮助
    """
    # 存储调试标志到上下文
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        import logging

        from ..utils.logger import get_logger

        logger = get_logger()
        logger.setLevel(logging.DEBUG)


# 注册子命令
cli.add_command(compress)
cli.add_command(decompress)
cli.add_command(verify)
cli.add_command(benchmark)


def main() -> None:
    """主入口函数（用于脚本调用）"""
    cli()


if __name__ == "__main__":
    main()
