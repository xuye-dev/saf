"""
测试报告生成器模块

提供多种格式的测试报告生成功能：
- CSV 格式报告
- JSON 格式报告
- Markdown 格式报告
- 控制台格式化输出

Author: 徐野
Date: 2025-11-23
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logger import get_logger
from ..utils.timer import format_time
from .benchmark import BenchmarkSummary

logger = get_logger(__name__)


class ReportGenerator:
    """
    测试报告生成器

    支持多种输出格式：
    - CSV（适合数据分析）
    - JSON（适合程序处理）
    - Markdown（适合文档展示）
    - 控制台输出（适合实时查看）

    Example:
        >>> generator = ReportGenerator()
        >>> generator.generate_csv(summary, "report.csv")
        >>> generator.generate_json(summary, "report.json")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录（默认为 experiments/benchmarks/reports）
        """
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/benchmarks/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"报告生成器已初始化，输出目录: {self.output_dir}")

    def generate_csv(
        self,
        summary: BenchmarkSummary,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        生成 CSV 格式的测试报告

        Args:
            summary: 基准测试汇总结果
            output_path: 输出文件路径（可选）

        Returns:
            Path: 生成的报告文件路径

        Example:
            >>> generator.generate_csv(summary, "benchmark_report.csv")
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"benchmark_report_{timestamp}.csv"
        else:
            output_path = Path(output_path)

        # 准备 CSV 字段
        fieldnames = [
            "name",
            "original_size",
            "compressed_size",
            "compression_ratio",
            "compression_time",
            "decompression_time",
            "peak_memory",
            "algorithm_type",
            "is_valid",
            "timestamp",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in summary.results:
                row = {
                    "name": result.name,
                    "original_size": result.original_size,
                    "compressed_size": result.compressed_size,
                    "compression_ratio": f"{result.compression_ratio:.4f}",
                    "compression_time": f"{result.compression_time:.6f}",
                    "decompression_time": f"{result.decompression_time:.6f}",
                    "peak_memory": result.peak_memory,
                    "algorithm_type": result.algorithm_type,
                    "is_valid": result.is_valid,
                    "timestamp": result.timestamp,
                }
                writer.writerow(row)

        logger.info(f"CSV 报告已生成: {output_path}")
        return output_path

    def generate_json(
        self,
        summary: BenchmarkSummary,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Path:
        """
        生成 JSON 格式的测试报告

        Args:
            summary: 基准测试汇总结果
            output_path: 输出文件路径（可选）
            indent: JSON 缩进空格数

        Returns:
            Path: 生成的报告文件路径

        Example:
            >>> generator.generate_json(summary, "benchmark_report.json")
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        else:
            output_path = Path(output_path)

        report_data = {
            "report_type": "benchmark",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_tests": summary.total_tests,
                "passed_tests": summary.passed_tests,
                "failed_tests": summary.failed_tests,
                "avg_compression_ratio": summary.avg_compression_ratio,
                "avg_compression_time": summary.avg_compression_time,
                "avg_decompression_time": summary.avg_decompression_time,
                "total_original_size": summary.total_original_size,
                "total_compressed_size": summary.total_compressed_size,
            },
            "results": [r.to_dict() for r in summary.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=indent, ensure_ascii=False)

        logger.info(f"JSON 报告已生成: {output_path}")
        return output_path

    def generate_markdown(
        self,
        summary: BenchmarkSummary,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "基准测试报告",
    ) -> Path:
        """
        生成 Markdown 格式的测试报告

        Args:
            summary: 基准测试汇总结果
            output_path: 输出文件路径（可选）
            title: 报告标题

        Returns:
            Path: 生成的报告文件路径

        Example:
            >>> generator.generate_markdown(summary, "benchmark_report.md")
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"benchmark_report_{timestamp}.md"
        else:
            output_path = Path(output_path)

        lines: List[str] = []

        # 标题
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 汇总信息
        lines.append("## 汇总")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 总测试数 | {summary.total_tests} |")
        lines.append(f"| 通过 | {summary.passed_tests} |")
        lines.append(f"| 失败 | {summary.failed_tests} |")
        lines.append(f"| 平均压缩比 | {summary.avg_compression_ratio:.2f}x |")
        lines.append(f"| 平均压缩时间 | {format_time(summary.avg_compression_time)} |")
        lines.append(f"| 平均解压时间 | {format_time(summary.avg_decompression_time)} |")
        lines.append(
            f"| 总原始大小 | {summary.total_original_size / 1024 / 1024:.2f} MB |"
        )
        lines.append(
            f"| 总压缩大小 | {summary.total_compressed_size / 1024 / 1024:.2f} MB |"
        )
        overall_ratio = summary.total_original_size / max(summary.total_compressed_size, 1)
        lines.append(f"| 总体压缩比 | {overall_ratio:.2f}x |")
        lines.append("")

        # 详细结果
        lines.append("## 详细结果")
        lines.append("")
        lines.append(
            "| 名称 | 原始大小 | 压缩大小 | 压缩比 | 压缩时间 | 解压时间 | 算法 | 状态 |"
        )
        lines.append("|------|---------|---------|-------|---------|---------|------|------|")

        for result in summary.results:
            status = "✅" if result.is_valid else "❌"
            original_size = self._format_size(result.original_size)
            compressed_size = self._format_size(result.compressed_size)
            compression_time = format_time(result.compression_time)
            decompression_time = format_time(result.decompression_time)

            lines.append(
                f"| {result.name} | {original_size} | {compressed_size} | "
                f"{result.compression_ratio:.2f}x | {compression_time} | "
                f"{decompression_time} | {result.algorithm_type} | {status} |"
            )

        lines.append("")

        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown 报告已生成: {output_path}")
        return output_path

    def print_console(
        self,
        summary: BenchmarkSummary,
        verbose: bool = False,
    ) -> str:
        """
        生成控制台格式化输出

        Args:
            summary: 基准测试汇总结果
            verbose: 是否显示详细信息

        Returns:
            str: 格式化的输出字符串
        """
        lines: List[str] = []

        # 分隔线
        sep = "=" * 60

        lines.append(sep)
        lines.append("              基准测试报告")
        lines.append(sep)
        lines.append("")

        # 汇总
        lines.append(f"总测试数:     {summary.total_tests}")
        lines.append(f"通过:         {summary.passed_tests}")
        lines.append(f"失败:         {summary.failed_tests}")
        lines.append(f"平均压缩比:   {summary.avg_compression_ratio:.2f}x")
        lines.append(f"平均压缩时间: {format_time(summary.avg_compression_time)}")
        lines.append(f"平均解压时间: {format_time(summary.avg_decompression_time)}")
        lines.append("")

        if verbose and summary.results:
            lines.append("-" * 60)
            lines.append("详细结果:")
            lines.append("-" * 60)

            for result in summary.results:
                status = "✓" if result.is_valid else "✗"
                lines.append(f"  [{status}] {result.name}")
                lines.append(f"      压缩比: {result.compression_ratio:.2f}x")
                lines.append(f"      算法: {result.algorithm_type}")
                lines.append(f"      压缩: {format_time(result.compression_time)}")
                lines.append(f"      解压: {format_time(result.decompression_time)}")
                lines.append("")

        lines.append(sep)

        output = "\n".join(lines)
        print(output)
        return output

    def generate_all(
        self,
        summary: BenchmarkSummary,
        base_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        生成所有格式的报告

        Args:
            summary: 基准测试汇总结果
            base_name: 基础文件名（可选）

        Returns:
            Dict[str, Path]: 格式名称到文件路径的映射
        """
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benchmark_report_{timestamp}"

        results = {
            "csv": self.generate_csv(
                summary, self.output_dir / f"{base_name}.csv"
            ),
            "json": self.generate_json(
                summary, self.output_dir / f"{base_name}.json"
            ),
            "markdown": self.generate_markdown(
                summary, self.output_dir / f"{base_name}.md"
            ),
        }

        logger.info(f"已生成所有格式报告: {list(results.keys())}")
        return results

    @staticmethod
    def _format_size(size_bytes: int) -> str:
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


class VerificationReportGenerator:
    """
    验证报告生成器

    专门用于生成数据验证报告

    Example:
        >>> from src.models import VerificationResult
        >>> generator = VerificationReportGenerator()
        >>> generator.generate_report(results, "verification_report.json")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        初始化验证报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/benchmarks/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        format: str = "json",
    ) -> Path:
        """
        生成验证报告

        Args:
            results: 验证结果字典（文件路径 -> VerificationResult）
            output_path: 输出文件路径
            format: 输出格式 ("json" 或 "csv")

        Returns:
            Path: 生成的报告文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"verification_report_{timestamp}.{format}"
        else:
            output_path = Path(output_path)

        # 统计
        total = len(results)
        passed = sum(1 for r in results.values() if hasattr(r, "is_valid") and r.is_valid)
        failed = total - passed

        if format == "json":
            report_data = {
                "report_type": "verification",
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                },
                "results": {
                    path: {
                        "is_valid": r.is_valid if hasattr(r, "is_valid") else False,
                        "original_hash": r.original_hash if hasattr(r, "original_hash") else "",
                        "reconstructed_hash": (
                            r.reconstructed_hash if hasattr(r, "reconstructed_hash") else ""
                        ),
                        "error_message": r.error_message if hasattr(r, "error_message") else None,
                        "verification_time": (
                            r.verification_time if hasattr(r, "verification_time") else 0.0
                        ),
                    }
                    for path, r in results.items()
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            fieldnames = [
                "file_path",
                "is_valid",
                "original_hash",
                "reconstructed_hash",
                "error_message",
                "verification_time",
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for path, r in results.items():
                    row = {
                        "file_path": path,
                        "is_valid": r.is_valid if hasattr(r, "is_valid") else False,
                        "original_hash": r.original_hash if hasattr(r, "original_hash") else "",
                        "reconstructed_hash": (
                            r.reconstructed_hash if hasattr(r, "reconstructed_hash") else ""
                        ),
                        "error_message": r.error_message if hasattr(r, "error_message") else "",
                        "verification_time": (
                            f"{r.verification_time:.6f}"
                            if hasattr(r, "verification_time")
                            else "0.0"
                        ),
                    }
                    writer.writerow(row)

        logger.info(f"验证报告已生成: {output_path}")
        return output_path


def generate_benchmark_report(
    summary: BenchmarkSummary,
    output_dir: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    生成基准测试报告（便捷函数）

    Args:
        summary: 基准测试汇总结果
        output_dir: 输出目录
        formats: 输出格式列表（默认 ["csv", "json", "markdown"]）

    Returns:
        Dict[str, Path]: 格式名称到文件路径的映射

    Example:
        >>> paths = generate_benchmark_report(summary, formats=["json", "csv"])
    """
    generator = ReportGenerator(output_dir=output_dir)

    if formats is None:
        return generator.generate_all(summary)

    results: Dict[str, Path] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"benchmark_report_{timestamp}"

    for fmt in formats:
        if fmt == "csv":
            results["csv"] = generator.generate_csv(
                summary, generator.output_dir / f"{base_name}.csv"
            )
        elif fmt == "json":
            results["json"] = generator.generate_json(
                summary, generator.output_dir / f"{base_name}.json"
            )
        elif fmt == "markdown":
            results["markdown"] = generator.generate_markdown(
                summary, generator.output_dir / f"{base_name}.md"
            )

    return results
