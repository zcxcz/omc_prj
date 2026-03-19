#!/usr/bin/env python3
"""
CSIIR Pattern Validation Runner

Unified script to run both Python and C++ models with pattern output
and compare the results.

Usage:
    python3 run_pattern_validation.py --pattern flat --size 64x64 --pixel-bits 10

Features:
    - Generates test patterns
    - Runs Python reference model with pattern output
    - Runs C++ HLS model with pattern output
    - Compares all intermediate pipeline stages
    - Generates detailed comparison report

Author: HLS Team
Date: 2026-03-18
Version: 1.0
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

# Add algorithms directory to path
sys.path.insert(0, str(Path(__file__).parent))

from csiir_pattern_output import PatternOutputCSIIR
from csiir_pattern_compare import generate_comparison_report
from csiir_c2c_utils import generate_test_pattern, save_binary


def main():
    parser = argparse.ArgumentParser(
        description='CSIIR Pattern Validation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single pattern validation
  python3 run_pattern_validation.py --pattern flat --size 64x64

  # Run all patterns
  python3 run_pattern_validation.py --pattern all --size 32x32 --pixel-bits 10

  # Custom output directory
  python3 run_pattern_validation.py --pattern gradient --output-dir ./pattern_data
        """
    )

    parser.add_argument('--pattern', type=str, default='flat',
                       help='Pattern name (flat, gradient, edge, checkerboard, noise, natural, or "all")')
    parser.add_argument('--size', type=str, default='64x64',
                       help='Image size WxH')
    parser.add_argument('--pixel-bits', type=int, default=10, choices=[8, 10, 12],
                       help='Pixel bit width')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory')
    parser.add_argument('--skip-cpp', action='store_true',
                       help='Skip C++ model execution')
    parser.add_argument('--compile-cpp', action='store_true',
                       help='Force recompile C++ testbench')

    args = parser.parse_args()

    # Parse size
    try:
        width, height = map(int, args.size.lower().split('x'))
    except ValueError:
        print(f"Error: Invalid size format '{args.size}'. Use WxH format.")
        return 1

    # Determine output directory
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).parent.parent / "hls_csiir" / "pattern_data"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine patterns
    if args.pattern.lower() == 'all':
        patterns = ['flat', 'gradient', 'edge', 'checkerboard', 'noise', 'natural']
    else:
        patterns = [args.pattern]

    pixel_max = (1 << args.pixel_bits) - 1
    scale = (1 << args.pixel_bits) // 256
    thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale
    blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

    # Project paths
    project_root = Path(__file__).parent.parent
    hls_dir = project_root / "hls_csiir"
    # Use full pipeline testbench (calls complete HLS modules)
    binary_name = f"tb_csiir_pattern_full_{args.pixel_bits}bit"
    binary_path = hls_dir / binary_name

    # Compile C++ if needed
    if not args.skip_cpp and (args.compile_cpp or not binary_path.exists()):
        print(f"\n{'='*70}")
        print("Compiling C++ Pattern Testbench")
        print(f"{'='*70}")

        cmd = [
            'make', '-C', str(hls_dir),
            f'pattern-full-{args.pixel_bits}bit'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return 1
        print(f"Compiled: {binary_path}")

    # Process each pattern
    for pattern_name in patterns:
        print(f"\n{'='*70}")
        print(f"Pattern: {pattern_name} ({width}x{height}, {args.pixel_bits}-bit)")
        print(f"{'='*70}")

        pattern_dir_name = f"{pattern_name}_{width}x{height}_{args.pixel_bits}bit"
        python_dir = base_dir / f"{pattern_dir_name}_python"
        cpp_dir = base_dir / f"{pattern_dir_name}_cpp"

        # Generate test pattern
        print("\n[1/4] Generating test pattern...")
        y, u, v = generate_test_pattern(pattern_name, height, width, pixel_max=pixel_max)

        # Run Python model
        print("\n[2/4] Running Python model with pattern output...")
        python_dir.mkdir(parents=True, exist_ok=True)
        csiir = PatternOutputCSIIR(pixel_bits=args.pixel_bits)
        csiir.save_pattern_data(y, u, v, str(python_dir), pattern_name, thresh, blend_ratio)

        # Run C++ model
        if not args.skip_cpp and binary_path.exists():
            print("\n[3/4] Running C++ model with pattern output...")
            cpp_dir.mkdir(parents=True, exist_ok=True)

            # Create input binary
            input_bin = cpp_dir / "input.bin"
            save_binary(str(input_bin), np.stack([y, u, v], axis=-1),
                       pixel_bits=args.pixel_bits, channels=3)

            # Run C++ testbench
            cmd = [str(binary_path), str(input_bin), str(cpp_dir), str(width), str(height)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"C++ execution failed:\n{result.stderr}")
            else:
                print(result.stdout)

            # Compare results
            print("\n[4/4] Comparing patterns...")
            report_file = base_dir / f"comparison_{pattern_dir_name}.txt"
            generate_comparison_report(
                str(python_dir), str(cpp_dir),
                str(report_file),
                verbose=True
            )
        else:
            print("\n[3/4] Skipping C++ model (--skip-cpp or binary not found)")
            print("\n[4/4] Comparison skipped")

    print(f"\n{'='*70}")
    print("Pattern validation completed!")
    print(f"Output directory: {base_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())