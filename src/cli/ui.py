"""
CLI UI 工具模块

提供命令行界面的辅助功能：
- 彩色输出
- 进度条显示
- 格式化输出

Author: 徐野
Date: 2025-11-23
"""

import sys
from typing import Any, Optional

import click
from tqdm import tqdm


# ===== 颜色样式常量 =====
class Style:
    """输出样式常量"""

    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    INFO = "blue"
    HIGHLIGHT = "cyan"
    DIM = "bright_black"


# ===== 输出函数 =====
def echo_success(message: str) -> None:
    """输出成功信息（绿色）"""
    click.echo(click.style(f"✓ {message}", fg=Style.SUCCESS))


def echo_error(message: str) -> None:
    """输出错误信息（红色）"""
    click.echo(click.style(f"✗ {message}", fg=Style.ERROR), err=True)


def echo_warning(message: str) -> None:
    """输出警告信息（黄色）"""
    click.echo(click.style(f"⚠ {message}", fg=Style.WARNING))


def echo_info(message: str) -> None:
    """输出提示信息（蓝色）"""
    click.echo(click.style(f"ℹ {message}", fg=Style.INFO))


def echo_highlight(message: str) -> None:
    """输出高亮信息（青色）"""
    click.echo(click.style(message, fg=Style.HIGHLIGHT))


def echo_dim(message: str) -> None:
    """输出暗淡信息（灰色）"""
    click.echo(click.style(message, fg=Style.DIM))


# ===== 格式化输出 =====
def format_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        str: 格式化的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def format_ratio(ratio: float) -> str:
    """
    格式化压缩比

    Args:
        ratio: 压缩比

    Returns:
        str: 格式化的压缩比字符串
    """
    if ratio >= 1000:
        return f"{ratio:.0f}x"
    elif ratio >= 100:
        return f"{ratio:.1f}x"
    else:
        return f"{ratio:.2f}x"


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ===== 进度条 =====
def create_progress_bar(
    total: int,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
) -> tqdm:
    """
    创建进度条

    Args:
        total: 总数
        desc: 描述文本
        unit: 单位
        disable: 是否禁用

    Returns:
        tqdm: 进度条对象
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        file=sys.stderr,
    )


# ===== 表格输出 =====
def print_table(
    headers: list[str],
    rows: list[list[Any]],
    title: Optional[str] = None,
) -> None:
    """
    打印格式化表格

    Args:
        headers: 表头列表
        rows: 数据行列表
        title: 表格标题（可选）
    """
    # 计算每列宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 生成分隔线
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # 打印标题
    if title:
        click.echo()
        click.echo(click.style(title, fg=Style.HIGHLIGHT, bold=True))

    # 打印表头
    click.echo(separator)
    header_row = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    click.echo(click.style(header_row, bold=True))
    click.echo(separator)

    # 打印数据行
    for row in rows:
        row_str = (
            "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        )
        click.echo(row_str)

    click.echo(separator)


# ===== 结果摘要 =====
def print_compression_result(
    input_file: str,
    output_file: str,
    original_size: int,
    compressed_size: int,
    compression_ratio: float,
    algorithm_type: str,
    compression_time: float,
) -> None:
    """
    打印压缩结果摘要

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        original_size: 原始大小
        compressed_size: 压缩后大小
        compression_ratio: 压缩比
        algorithm_type: 算法类型
        compression_time: 压缩耗时
    """
    click.echo()
    click.echo(click.style("压缩完成", fg=Style.SUCCESS, bold=True))
    click.echo(click.style("─" * 40, fg=Style.DIM))
    click.echo(f"  输入文件: {input_file}")
    click.echo(f"  输出文件: {output_file}")
    click.echo(f"  原始大小: {format_size(original_size)}")
    click.echo(f"  压缩大小: {format_size(compressed_size)}")
    click.echo(
        f"  压缩比:   {click.style(format_ratio(compression_ratio), fg=Style.HIGHLIGHT, bold=True)}"
    )
    click.echo(f"  算法类型: {algorithm_type}")
    click.echo(f"  耗时:     {format_time(compression_time)}")
    click.echo()


def print_decompression_result(
    input_file: str,
    output_file: str,
    is_valid: bool,
    decompression_time: float,
) -> None:
    """
    打印解压结果摘要

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        is_valid: 验证是否通过
        decompression_time: 解压耗时
    """
    click.echo()
    if is_valid:
        click.echo(click.style("解压完成", fg=Style.SUCCESS, bold=True))
    else:
        click.echo(click.style("解压完成（验证失败）", fg=Style.WARNING, bold=True))
    click.echo(click.style("─" * 40, fg=Style.DIM))
    click.echo(f"  输入文件: {input_file}")
    click.echo(f"  输出文件: {output_file}")
    status = (
        click.style("✓ 通过", fg=Style.SUCCESS)
        if is_valid
        else click.style("✗ 失败", fg=Style.ERROR)
    )
    click.echo(f"  哈希验证: {status}")
    click.echo(f"  耗时:     {format_time(decompression_time)}")
    click.echo()


def print_verification_result(
    file_path: str,
    is_valid: bool,
    original_hash: str,
    reconstructed_hash: str,
    verification_time: float,
    error_message: Optional[str] = None,
) -> None:
    """
    打印验证结果摘要

    Args:
        file_path: 文件路径
        is_valid: 验证是否通过
        original_hash: 原始哈希
        reconstructed_hash: 重建哈希
        verification_time: 验证耗时
        error_message: 错误信息（可选）
    """
    click.echo()
    if is_valid:
        click.echo(click.style("验证通过", fg=Style.SUCCESS, bold=True))
    else:
        click.echo(click.style("验证失败", fg=Style.ERROR, bold=True))
    click.echo(click.style("─" * 40, fg=Style.DIM))
    click.echo(f"  文件:     {file_path}")
    click.echo(f"  原始哈希: {original_hash[:16]}...")
    click.echo(f"  重建哈希: {reconstructed_hash[:16]}...")
    click.echo(f"  耗时:     {format_time(verification_time)}")
    if error_message:
        click.echo(f"  错误:     {click.style(error_message, fg=Style.ERROR)}")
    click.echo()
