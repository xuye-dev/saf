#!/usr/bin/env python3
"""
生成器验证脚本 - 生成示例数据并保存到 data/ 目录

Author: 徐野
Date: 2025-11-23
"""

from pathlib import Path

from src.generators import (
    CheckerboardGenerator,
    FibonacciGenerator,
    JuliaGenerator,
    MandelbrotGenerator,
    PerlinNoiseGenerator,
    PiDigitsGenerator,
    PrimeGenerator,
    StripeGenerator,
    WhiteNoiseGenerator,
)
from src.utils import format_file_size, get_file_size


def generate_sequences() -> None:
    """生成数学序列数据"""
    print("\n=== 生成数学序列数据 ===")
    sequences_dir = Path("data/sequences")
    sequences_dir.mkdir(parents=True, exist_ok=True)

    # 1. 斐波那契数列（前 10000 项）
    print("1. 生成斐波那契数列（前 10000 项）...")
    fib_gen = FibonacciGenerator()
    fib_data = fib_gen.generate(n=10000)
    fib_path = sequences_dir / "fibonacci_10000.npy"
    fib_gen.save_npy(fib_data, fib_path)
    print(f"   ✓ 已保存到: {fib_path}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(fib_path))}")
    print(f"   ✓ 前 10 项: {fib_data[:10].tolist()}")

    # 2. 素数序列（前 10000 个素数）
    print("\n2. 生成素数序列（前 10000 个素数）...")
    prime_gen = PrimeGenerator()
    prime_data = prime_gen.generate(n=10000)
    prime_path = sequences_dir / "primes_10000.npy"
    prime_gen.save_npy(prime_data, prime_path)
    print(f"   ✓ 已保存到: {prime_path}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(prime_path))}")
    print(f"   ✓ 前 10 个素数: {prime_data[:10].tolist()}")

    # 3. π 的位数（10000 位）
    print("\n3. 生成 π 的位数（10000 位）...")
    pi_gen = PiDigitsGenerator()
    pi_data = pi_gen.generate(n=10000, include_decimal_point=True)
    pi_path = sequences_dir / "pi_digits_10000.npy"
    pi_gen.save_npy(pi_data, pi_path)
    print(f"   ✓ 已保存到: {pi_path}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(pi_path))}")
    print(f"   ✓ 前 20 位: {pi_data[:20].tolist()}")

    # 同时保存文本格式（便于查看）
    pi_txt_path = sequences_dir / "pi_digits_10000.txt"
    pi_gen.save_txt(pi_data, pi_txt_path, fmt="%d")
    print(f"   ✓ 文本格式: {pi_txt_path}")


def generate_fractals() -> None:
    """生成分形图像数据"""
    print("\n\n=== 生成分形图像数据 ===")
    fractals_dir = Path("data/fractals")
    fractals_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mandelbrot 集合（800x600，256 次迭代）
    print("1. 生成 Mandelbrot 分形（800x600，256 次迭代）...")
    mandelbrot_gen = MandelbrotGenerator()
    mandelbrot_data = mandelbrot_gen.generate(width=800, height=600, max_iter=256)
    mandelbrot_npy = fractals_dir / "mandelbrot_800x600.npy"
    mandelbrot_gen.save_npy(mandelbrot_data, mandelbrot_npy)
    print(f"   ✓ 已保存到: {mandelbrot_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(mandelbrot_npy))}")

    # 保存为 PNG 图像（多种颜色映射）
    mandelbrot_png_hot = fractals_dir / "mandelbrot_800x600_hot.png"
    mandelbrot_gen.save_png(mandelbrot_data, mandelbrot_png_hot, colormap="hot")
    print(f"   ✓ PNG (hot):  {mandelbrot_png_hot}")

    mandelbrot_png_viridis = fractals_dir / "mandelbrot_800x600_viridis.png"
    mandelbrot_gen.save_png(mandelbrot_data, mandelbrot_png_viridis, colormap="viridis")
    print(f"   ✓ PNG (viridis): {mandelbrot_png_viridis}")

    # 2. Julia 集合（800x600，256 次迭代）
    print("\n2. 生成 Julia 分形（800x600，256 次迭代）...")
    julia_gen = JuliaGenerator(c=-0.7 + 0.27015j)
    julia_data = julia_gen.generate(width=800, height=600, max_iter=256)
    julia_npy = fractals_dir / "julia_800x600.npy"
    julia_gen.save_npy(julia_data, julia_npy)
    print(f"   ✓ 已保存到: {julia_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(julia_npy))}")

    # 保存为 PNG 图像
    julia_png = fractals_dir / "julia_800x600_viridis.png"
    julia_gen.save_png(julia_data, julia_png, colormap="viridis")
    print(f"   ✓ PNG: {julia_png}")

    # 3. 高分辨率 Mandelbrot（1920x1080，512 次迭代）
    print("\n3. 生成高分辨率 Mandelbrot 分形（1920x1080，512 次迭代）...")
    mandelbrot_hd_gen = MandelbrotGenerator()
    mandelbrot_hd_data = mandelbrot_hd_gen.generate(width=1920, height=1080, max_iter=512)
    mandelbrot_hd_npy = fractals_dir / "mandelbrot_1920x1080_hd.npy"
    mandelbrot_hd_gen.save_npy(mandelbrot_hd_data, mandelbrot_hd_npy)
    print(f"   ✓ 已保存到: {mandelbrot_hd_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(mandelbrot_hd_npy))}")

    mandelbrot_hd_png = fractals_dir / "mandelbrot_1920x1080_hd.png"
    mandelbrot_hd_gen.save_png(mandelbrot_hd_data, mandelbrot_hd_png, colormap="hot")
    print(f"   ✓ PNG: {mandelbrot_hd_png}")


def generate_noise_and_patterns() -> None:
    """生成噪声和图案数据"""
    print("\n\n=== 生成噪声和图案数据 ===")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Perlin 噪声（512x512）
    print("1. 生成 Perlin 噪声（512x512）...")
    perlin_gen = PerlinNoiseGenerator(seed=42)
    perlin_data = perlin_gen.generate(width=512, height=512, scale=100.0)
    perlin_npy = results_dir / "perlin_noise_512x512.npy"
    perlin_gen.save_npy(perlin_data, perlin_npy)
    print(f"   ✓ 已保存到: {perlin_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(perlin_npy))}")

    perlin_png = results_dir / "perlin_noise_512x512.png"
    perlin_gen.save_png(perlin_data, perlin_png)
    print(f"   ✓ PNG: {perlin_png}")

    # 2. 白噪声（512x512）
    print("\n2. 生成白噪声（512x512，均匀分布）...")
    white_noise_gen = WhiteNoiseGenerator(seed=42)
    white_noise_data = white_noise_gen.generate(width=512, height=512, distribution="uniform")
    white_noise_npy = results_dir / "white_noise_512x512.npy"
    white_noise_gen.save_npy(white_noise_data, white_noise_npy)
    print(f"   ✓ 已保存到: {white_noise_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(white_noise_npy))}")

    white_noise_png = results_dir / "white_noise_512x512.png"
    white_noise_gen.save_png(white_noise_data, white_noise_png)
    print(f"   ✓ PNG: {white_noise_png}")

    # 3. 棋盘图案（512x512）
    print("\n3. 生成棋盘图案（512x512，格子大小 32）...")
    checkerboard_gen = CheckerboardGenerator()
    checkerboard_data = checkerboard_gen.generate(width=512, height=512, square_size=32)
    checkerboard_npy = results_dir / "checkerboard_512x512.npy"
    checkerboard_gen.save_npy(checkerboard_data, checkerboard_npy)
    print(f"   ✓ 已保存到: {checkerboard_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(checkerboard_npy))}")

    checkerboard_png = results_dir / "checkerboard_512x512.png"
    checkerboard_gen.save_png(checkerboard_data, checkerboard_png)
    print(f"   ✓ PNG: {checkerboard_png}")

    # 4. 条纹图案（512x512）
    print("\n4. 生成条纹图案（512x512，条纹宽度 16）...")
    stripe_gen = StripeGenerator()
    stripe_data = stripe_gen.generate(
        width=512, height=512, stripe_width=16, orientation="horizontal"
    )
    stripe_npy = results_dir / "stripes_512x512.npy"
    stripe_gen.save_npy(stripe_data, stripe_npy)
    print(f"   ✓ 已保存到: {stripe_npy}")
    print(f"   ✓ 文件大小: {format_file_size(get_file_size(stripe_npy))}")

    stripe_png = results_dir / "stripes_512x512.png"
    stripe_gen.save_png(stripe_data, stripe_png)
    print(f"   ✓ PNG: {stripe_png}")


def main() -> None:
    """主函数"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "数据生成器验证脚本" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        # 生成数学序列
        generate_sequences()

        # 生成分形图像
        generate_fractals()

        # 生成噪声和图案
        generate_noise_and_patterns()

        print("\n\n" + "=" * 80)
        print("✅ 所有数据生成完毕！")
        print("=" * 80)
        print("\n📁 数据目录:")
        print("   - 序列数据: data/sequences/")
        print("   - 分形图像: data/fractals/")
        print("   - 噪声和图案: data/results/")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        raise


if __name__ == "__main__":
    main()
