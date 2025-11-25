"""
Generators Module - Sequence, Fractal, Noise, and Pattern Generators

Author: 徐野
Date: 2025-11-23
"""

# Base classes
from .base import BaseGenerator, ProgressCallback

# Fractal generators
from .fractal import JuliaGenerator, MandelbrotGenerator

# Noise and pattern generators
from .noise import (
    CheckerboardGenerator,
    PerlinNoiseGenerator,
    StripeGenerator,
    WhiteNoiseGenerator,
)

# Sequence generators
from .sequence import (
    FibonacciGenerator,
    FormulaGenerator,
    PiDigitsGenerator,
    PrimeGenerator,
)

__all__ = [
    # Base classes
    "BaseGenerator",
    "ProgressCallback",
    # Sequence generators
    "FibonacciGenerator",
    "PrimeGenerator",
    "PiDigitsGenerator",
    "FormulaGenerator",
    # Fractal generators
    "MandelbrotGenerator",
    "JuliaGenerator",
    # Noise and pattern generators
    "PerlinNoiseGenerator",
    "WhiteNoiseGenerator",
    "CheckerboardGenerator",
    "StripeGenerator",
]
