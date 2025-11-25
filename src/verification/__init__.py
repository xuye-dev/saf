"""
Verification module for data integrity and performance testing.

Provides:
- Lossless verification (hash verification, byte-by-byte comparison)
- Performance benchmarking (compression/decompression time, memory monitoring)
- Test report generation (CSV, JSON, Markdown)

Author: 徐野
Date: 2025-11-23
"""

from .benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    MemoryProfiler,
    PerformanceBenchmark,
    benchmark_directory,
    benchmark_file,
)
from .report import (
    ReportGenerator,
    VerificationReportGenerator,
    generate_benchmark_report,
)
from .verifier import (
    BatchVerifier,
    DataVerifier,
    verify_data,
    verify_file,
)

__all__ = [
    # Verifier
    "DataVerifier",
    "BatchVerifier",
    "verify_data",
    "verify_file",
    # Benchmark
    "PerformanceBenchmark",
    "BenchmarkResult",
    "BenchmarkSummary",
    "MemoryProfiler",
    "benchmark_file",
    "benchmark_directory",
    # Report
    "ReportGenerator",
    "VerificationReportGenerator",
    "generate_benchmark_report",
]
