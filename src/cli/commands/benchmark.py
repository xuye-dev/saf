"""
基准测试命令实现

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path
from typing import Optional

import click

from ...utils.logger import get_logger
from ...verification.benchmark import BenchmarkSummary, PerformanceBenchmark
from ...verification.report import ReportGenerator
from ..ui import (
    Style,
    echo_error,
    echo_info,
    echo_success,
    format_ratio,
    format_size,
    format_time,
    print_table,
)

logger = get_logger(__name__)


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-p",
    "--pattern",
    default="*.npy",
    show_default=True,
    help="文件匹配模式",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="报告输出目录",
)
@click.option(
    "-f",
    "--format",
    "report_format",
    type=click.Choice(["csv", "json", "md", "all"]),
    default=None,
    help="报告输出格式",
)
@click.option(
    "-t",
    "--threshold",
    "confidence_threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="置信度阈值",
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
    help="静默模式（仅输出摘要）",
)
def benchmark(
    directory: str,
    pattern: str,
    output_dir: Optional[str],
    report_format: Optional[str],
    confidence_threshold: float,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    对目录中的数据文件进行批量基准测试

    测试所有匹配的 .npy 文件的压缩/解压性能。

    \b
    示例:
      saf benchmark data/sequences/            # 测试目录中的所有 .npy 文件
      saf benchmark data/ -p "*.npy"           # 指定匹配模式
      saf benchmark data/ -f json              # 生成 JSON 报告
      saf benchmark data/ -f all -o reports/   # 生成所有格式报告
    """
    dir_path = Path(directory)

    if not quiet:
        echo_info(f"正在扫描目录: {dir_path}")
        echo_info(f"匹配模式: {pattern}")

    try:
        # 查找匹配的文件
        files = list(dir_path.glob(pattern))
        if not files:
            echo_error(f"未找到匹配的文件: {pattern}")
            raise SystemExit(1)

        if not quiet:
            echo_info(f"找到 {len(files)} 个文件")

        # 执行基准测试
        bench = PerformanceBenchmark(
            confidence_threshold=confidence_threshold,
            output_dir=output_dir or "experiments/benchmarks",
        )
        summary = bench.run_directory(dir_path, pattern=pattern, cleanup=True)

        # 打印结果表格
        if not quiet:
            _print_benchmark_summary(summary, verbose)

        # 生成报告
        if report_format:
            _generate_reports(summary, output_dir, report_format, quiet)

        # 设置退出码
        if summary.failed_tests > 0:
            raise SystemExit(1)

    except FileNotFoundError as e:
        echo_error(f"目录不存在: {e}")
        raise SystemExit(1)
    except Exception as e:
        echo_error(f"基准测试失败: {e}")
        logger.exception("基准测试过程中发生错误")
        raise SystemExit(1)


def _print_benchmark_summary(summary: "BenchmarkSummary", verbose: bool) -> None:
    """打印基准测试摘要"""
    click.echo()
    click.echo(click.style("═" * 60, fg=Style.DIM))
    click.echo(click.style("  基准测试结果", fg=Style.HIGHLIGHT, bold=True))
    click.echo(click.style("═" * 60, fg=Style.DIM))

    # 汇总信息
    click.echo()
    click.echo(f"  总测试数:     {summary.total_tests}")
    passed_style = Style.SUCCESS if summary.passed_tests == summary.total_tests else Style.WARNING
    click.echo(f"  通过:         {click.style(str(summary.passed_tests), fg=passed_style)}")
    if summary.failed_tests > 0:
        click.echo(f"  失败:         {click.style(str(summary.failed_tests), fg=Style.ERROR)}")
    ratio_str = format_ratio(summary.avg_compression_ratio)
    click.echo(
        f"  平均压缩比:   {click.style(ratio_str, fg=Style.HIGHLIGHT, bold=True)}"
    )
    click.echo(f"  平均压缩时间: {format_time(summary.avg_compression_time)}")
    click.echo(f"  平均解压时间: {format_time(summary.avg_decompression_time)}")
    click.echo(f"  总原始大小:   {format_size(summary.total_original_size)}")
    click.echo(f"  总压缩大小:   {format_size(summary.total_compressed_size)}")

    # 详细结果表格
    if verbose and summary.results:
        click.echo()
        headers = ["名称", "原始大小", "压缩大小", "压缩比", "算法", "状态"]
        rows = []
        for r in summary.results:
            status = "✓" if r.is_valid else "✗"
            rows.append(
                [
                    r.name[:20],
                    format_size(r.original_size),
                    format_size(r.compressed_size),
                    format_ratio(r.compression_ratio),
                    r.algorithm_type,
                    status,
                ]
            )
        print_table(headers, rows, title="详细结果")

    click.echo()


def _generate_reports(
    summary: "BenchmarkSummary", output_dir: Optional[str], report_format: str, quiet: bool
) -> None:
    """生成测试报告"""
    generator = ReportGenerator(output_dir=output_dir)

    if report_format == "all":
        paths = generator.generate_all(summary)
        if not quiet:
            echo_success("已生成所有格式报告:")
            for fmt, path in paths.items():
                echo_info(f"  {fmt}: {path}")
    else:
        if report_format == "csv":
            path = generator.generate_csv(summary)
        elif report_format == "json":
            path = generator.generate_json(summary)
        elif report_format == "md":
            path = generator.generate_markdown(summary)
        else:
            return

        if not quiet:
            echo_success(f"已生成报告: {path}")
