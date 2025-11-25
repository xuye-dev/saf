"""
Pattern Detector Module

Author: Xu Ye
Date: 2025-11-23
"""

from .base import (
    BaseDetector,
    ProgressCallback,
    calculate_r_squared,
    calculate_relative_error,
)
from .fractal_detector import FractalDetector
from .pattern_matcher import PatternMatcher
from .sequence_detector import SequenceDetector

__all__ = [
    "BaseDetector",
    "ProgressCallback",
    "calculate_r_squared",
    "calculate_relative_error",
    "SequenceDetector",
    "FractalDetector",
    "PatternMatcher",
]
