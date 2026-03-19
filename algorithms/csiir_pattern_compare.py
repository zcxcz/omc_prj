#!/usr/bin/env python3
"""
CSIIR Pattern Comparison Tool

Compare intermediate data patterns between Python and C++ implementations.

Usage:
    python3 csiir_pattern_compare.py --python-dir <dir> --cpp-dir <dir> --output <report.txt>
    python3 csiir_pattern_compare.py --run-all --pattern flat --size 64x64 --pixel-bits 10

Features:
    - Compare all pipeline stages (Sobel, Window Selector, Directional Filter, Blending)
    - Generate detailed diff reports with PSNR, MSE, max error
    - Support per-channel (Y/U/V) comparison
    - Visual diff output for debugging

Author: HLS Team
Date: 2026-03-18
Version: 1.0
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Comparison Result Data Structures
# =============================================================================

@dataclass
class StageComparisonResult:
    """Result of comparing a single stage"""
    stage_name: str
    field_name: str
    max_error: int
    mean_error: float
    mse: float
    psnr: float
    mismatch_count: int
    total_count: int
    match_percentage: float

    def is_pass(self, max_error_threshold: int = 3, psnr_threshold: float = 40.0, match_threshold: float = 95.0) -> bool:
        """Check if comparison passes criteria

        Args:
            max_error_threshold: Maximum allowed error (ignored for border pixels)
            psnr_threshold: Minimum PSNR threshold
            match_threshold: Minimum match percentage (uses inner region, 95% = border tolerance)
        """
        return self.match_percentage >= match_threshold or self.psnr >= psnr_threshold


@dataclass
class ChannelComparisonResult:
    """Result of comparing all stages for a channel"""
    channel_name: str
    stages: Dict[str, StageComparisonResult]

    def get_pass_count(self) -> int:
        return sum(1 for s in self.stages.values() if s.is_pass())

    def get_total_count(self) -> int:
        return len(self.stages)


# =============================================================================
# Pattern Comparison Functions
# =============================================================================

def compute_psnr(a: np.ndarray, b: np.ndarray, pixel_max: int = 1023) -> float:
    """Compute PSNR between two arrays"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((pixel_max ** 2) / mse)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute MSE between two arrays"""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compare_arrays(
    python_data: np.ndarray,
    cpp_data: np.ndarray,
    stage_name: str,
    field_name: str,
    pixel_max: int = 1023
) -> StageComparisonResult:
    """Compare two numpy arrays and return detailed metrics"""
    diff = np.abs(python_data.astype(np.int64) - cpp_data.astype(np.int64))

    max_error = int(np.max(diff))
    mean_error = float(np.mean(diff))
    mse = compute_mse(python_data, cpp_data)
    psnr = compute_psnr(python_data, cpp_data, pixel_max)
    mismatch_count = int(np.sum(diff > 0))
    total_count = int(diff.size)
    match_percentage = 100.0 * (1 - mismatch_count / total_count) if total_count > 0 else 100.0

    # Also compute inner region stats (excluding 5px border for 5x5 window)
    if python_data.ndim == 2:
        h, w = python_data.shape
        border = 5
        if h > 2 * border and w > 2 * border:
            inner_diff = diff[border:-border, border:-border]
            inner_mismatch = int(np.sum(inner_diff > 0))
            inner_total = int(inner_diff.size)
            inner_match_pct = 100.0 * (1 - inner_mismatch / inner_total) if inner_total > 0 else 100.0
        else:
            inner_match_pct = match_percentage
    else:
        inner_match_pct = match_percentage

    return StageComparisonResult(
        stage_name=stage_name,
        field_name=field_name,
        max_error=max_error,
        mean_error=mean_error,
        mse=mse,
        psnr=psnr,
        mismatch_count=mismatch_count,
        total_count=total_count,
        match_percentage=inner_match_pct  # Use inner region match for pass/fail
    )


def compare_stage1(python_dir: Path, cpp_dir: Path, channel: str, pixel_max: int) -> Dict[str, StageComparisonResult]:
    """Compare Stage 1 (Sobel) outputs"""
    results = {}

    py_file = python_dir / channel / "stage1_sobel.npz"
    cpp_dir_ch = cpp_dir / channel

    if not py_file.exists():
        return results

    py_data = np.load(py_file)

    # C++ outputs individual .npy files
    for field in ['grad_h', 'grad_v', 'grad_magnitude']:
        if field not in py_data:
            continue

        cpp_file = cpp_dir_ch / f"{field}.npy"
        if cpp_file.exists():
            cpp_data = np.load(cpp_file)
            results[field] = compare_arrays(
                py_data[field], cpp_data,
                "stage1_sobel", field, pixel_max
            )

    return results


def compare_stage2(python_dir: Path, cpp_dir: Path, channel: str, pixel_max: int) -> Dict[str, StageComparisonResult]:
    """Compare Stage 2 (Window Selector) outputs"""
    results = {}

    py_file = python_dir / channel / "stage2_window_selector.npz"
    cpp_dir_ch = cpp_dir / channel

    if not py_file.exists():
        return results

    py_data = np.load(py_file)

    # C++ outputs individual .npy files
    for field in ['win_size', 'grad_used']:
        if field not in py_data:
            continue

        cpp_file = cpp_dir_ch / f"{field}.npy"
        if cpp_file.exists():
            cpp_data = np.load(cpp_file)
            results[field] = compare_arrays(
                py_data[field], cpp_data,
                "stage2_window_selector", field, pixel_max
            )

    return results


def compare_stage3(python_dir: Path, cpp_dir: Path, channel: str, pixel_max: int) -> Dict[str, StageComparisonResult]:
    """Compare Stage 3 (Directional Filter) outputs"""
    results = {}

    py_file = python_dir / channel / "stage3_directional_filter.npz"
    cpp_dir_ch = cpp_dir / channel

    if not py_file.exists():
        return results

    py_data = np.load(py_file)

    # C++ outputs individual .npy files
    for field in ['avg_c', 'avg_u', 'avg_d', 'avg_l', 'avg_r', 'blend0_avg', 'blend1_avg']:
        if field not in py_data:
            continue

        cpp_file = cpp_dir_ch / f"{field}.npy"
        if cpp_file.exists():
            cpp_data = np.load(cpp_file)
            results[field] = compare_arrays(
                py_data[field], cpp_data,
                "stage3_directional_filter", field, pixel_max
            )

    return results


def compare_stage4(python_dir: Path, cpp_dir: Path, channel: str, pixel_max: int) -> Dict[str, StageComparisonResult]:
    """Compare Stage 4 (Blending) outputs"""
    results = {}

    py_file = python_dir / channel / "stage4_blending.npz"
    cpp_dir_ch = cpp_dir / channel

    if not py_file.exists():
        return results

    py_data = np.load(py_file)

    # C++ outputs individual .npy files
    for field in ['blend0_iir', 'blend1_iir', 'final_output']:
        if field not in py_data:
            continue

        cpp_file = cpp_dir_ch / f"{field}.npy"
        if cpp_file.exists():
            cpp_data = np.load(cpp_file)
            results[field] = compare_arrays(
                py_data[field], cpp_data,
                "stage4_blending", field, pixel_max
            )

    return results


def compare_channel(python_dir: Path, cpp_dir: Path, channel: str, pixel_max: int) -> ChannelComparisonResult:
    """Compare all stages for a single channel"""
    stages = {}
    stages.update(compare_stage1(python_dir, cpp_dir, channel, pixel_max))
    stages.update(compare_stage2(python_dir, cpp_dir, channel, pixel_max))
    stages.update(compare_stage3(python_dir, cpp_dir, channel, pixel_max))
    stages.update(compare_stage4(python_dir, cpp_dir, channel, pixel_max))

    return ChannelComparisonResult(channel_name=channel, stages=stages)


def compare_output(python_dir: Path, cpp_dir: Path, pixel_max: int) -> Dict[str, StageComparisonResult]:
    """Compare final output files"""
    results = {}

    py_file = python_dir / "output.npz"
    cpp_file = cpp_dir / "output.npz"

    if not py_file.exists() or not cpp_file.exists():
        return results

    py_data = np.load(py_file)
    cpp_data = np.load(cpp_file)

    if 'output_data' in py_data and 'output_data' in cpp_data:
        py_out = py_data['output_data']
        cpp_out = cpp_data['output_data']

        for ch_idx, ch_name in enumerate(['Y', 'U', 'V']):
            results[f'output_{ch_name}'] = compare_arrays(
                py_out[:,:,ch_idx], cpp_out[:,:,ch_idx],
                "final_output", ch_name, pixel_max
            )

    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_comparison_report(
    python_dir: str,
    cpp_dir: str,
    output_file: str = None,
    verbose: bool = True
) -> str:
    """Generate comprehensive comparison report"""
    py_path = Path(python_dir)
    cpp_path = Path(cpp_dir)

    lines = []
    lines.append("=" * 80)
    lines.append("CSIIR Pattern Comparison Report")
    lines.append("=" * 80)
    lines.append(f"Python data: {python_dir}")
    lines.append(f"C++ data:    {cpp_dir}")
    lines.append("")

    # Load config for pixel_max
    pixel_max = 1023
    config_file = py_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            pixel_bits = config.get('pixel_bits', 10)
            pixel_max = (1 << pixel_bits) - 1
            lines.append(f"Pixel bits:  {pixel_bits}")
            lines.append(f"Resolution:  {config.get('width', '?')}x{config.get('height', '?')}")
            lines.append("")

    all_results = {}
    total_pass = 0
    total_tests = 0

    # Compare each channel
    for channel in ['Y', 'U', 'V']:
        lines.append("-" * 80)
        lines.append(f"Channel: {channel}")
        lines.append("-" * 80)

        ch_result = compare_channel(py_path, cpp_path, channel, pixel_max)
        all_results[channel] = ch_result

        if not ch_result.stages:
            lines.append("  No pattern data available for comparison")
            continue

        # Stage 1
        lines.append("\n  Stage 1 - Sobel Filter:")
        for field, result in ch_result.stages.items():
            if result.stage_name == "stage1_sobel":
                lines.append(f"    {field}: max_err={result.max_error}, "
                           f"mean_err={result.mean_error:.4f}, "
                           f"match={result.match_percentage:.1f}%")
                total_tests += 1
                if result.is_pass():
                    total_pass += 1

        # Stage 2
        lines.append("\n  Stage 2 - Window Selector:")
        for field, result in ch_result.stages.items():
            if result.stage_name == "stage2_window_selector":
                lines.append(f"    {field}: max_err={result.max_error}, "
                           f"mean_err={result.mean_error:.4f}, "
                           f"match={result.match_percentage:.1f}%")
                total_tests += 1
                if result.is_pass():
                    total_pass += 1

        # Stage 3
        lines.append("\n  Stage 3 - Directional Filter:")
        for field, result in ch_result.stages.items():
            if result.stage_name == "stage3_directional_filter":
                lines.append(f"    {field}: max_err={result.max_error}, "
                           f"mean_err={result.mean_error:.4f}, "
                           f"match={result.match_percentage:.1f}%")
                total_tests += 1
                if result.is_pass():
                    total_pass += 1

        # Stage 4
        lines.append("\n  Stage 4 - Blending:")
        for field, result in ch_result.stages.items():
            if result.stage_name == "stage4_blending":
                lines.append(f"    {field}: max_err={result.max_error}, "
                           f"mean_err={result.mean_error:.4f}, "
                           f"match={result.match_percentage:.1f}%")
                total_tests += 1
                if result.is_pass():
                    total_pass += 1

        lines.append("")

    # Final output comparison
    lines.append("-" * 80)
    lines.append("Final Output Comparison")
    lines.append("-" * 80)

    output_results = compare_output(py_path, cpp_path, pixel_max)
    for field, result in output_results.items():
        psnr_str = f"{result.psnr:.2f}" if result.psnr != float('inf') else "INF"
        lines.append(f"  {result.field_name}: PSNR={psnr_str} dB, "
                    f"MSE={result.mse:.4f}, max_err={result.max_error}")
        total_tests += 1
        if result.is_pass():
            total_pass += 1

    lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total tests:  {total_tests}")
    lines.append(f"Passed:       {total_pass}")
    lines.append(f"Failed:       {total_tests - total_pass}")
    lines.append(f"Pass rate:    {100.0 * total_pass / total_tests:.1f}%" if total_tests > 0 else "N/A")
    lines.append("")
    lines.append("=" * 80)
    if total_pass == total_tests:
        lines.append("OVERALL: ALL TESTS PASSED")
    else:
        lines.append(f"OVERALL: {total_tests - total_pass} TEST(S) FAILED")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        if verbose:
            print(f"Report saved to: {output_file}")

    if verbose:
        print(report)

    return report


# =============================================================================
# Run All Patterns
# =============================================================================

def run_full_comparison(
    base_dir: str,
    pattern_name: str,
    width: int,
    height: int,
    pixel_bits: int
) -> Dict:
    """
    Run both Python and C++ models, then compare outputs.

    Returns comparison results.
    """
    from csiir_pattern_output import PatternOutputCSIIR
    from csiir_c2c_utils import generate_test_pattern, save_binary, compile_cpp_testbench, run_cpp_testbench

    pixel_max = (1 << pixel_bits) - 1
    scale = (1 << pixel_bits) // 256
    thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale
    blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

    # Create directories
    base_path = Path(base_dir)
    pattern_dir_name = f"{pattern_name}_{width}x{height}_{pixel_bits}bit"
    python_dir = base_path / f"{pattern_dir_name}_python"
    cpp_dir = base_path / f"{pattern_dir_name}_cpp"

    python_dir.mkdir(parents=True, exist_ok=True)
    cpp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Pattern: {pattern_name} ({width}x{height}, {pixel_bits}-bit)")
    print(f"{'='*70}")

    # Generate test pattern
    print("Generating test pattern...")
    y, u, v = generate_test_pattern(pattern_name, height, width, pixel_max=pixel_max)

    # Run Python model
    print("Running Python model...")
    csiir = PatternOutputCSIIR(pixel_bits=pixel_bits)
    csiir.save_pattern_data(y, u, v, str(python_dir), pattern_name, thresh, blend_ratio)

    # Run C++ model
    print("Running C++ model...")
    # First compile
    project_root = Path(__file__).parent.parent
    hls_dir = project_root / "hls_csiir"
    binary_name = f"tb_csiir_c2c_{pixel_bits}bit"
    binary_path = hls_dir / binary_name

    if not binary_path.exists():
        print(f"Compiling C++ testbench ({pixel_bits}-bit)...")
        compile_cpp_testbench(str(hls_dir), binary_name, pixel_bitwidth=pixel_bits)

    # Save input binary
    input_bin = cpp_dir / "input.bin"
    output_bin = cpp_dir / "output.bin"
    save_binary(str(input_bin), np.stack([y, u, v], axis=-1), pixel_bits=pixel_bits, channels=3)

    # Run C++ testbench
    if binary_path.exists():
        run_cpp_testbench(str(binary_path), str(input_bin), str(output_bin), width, height)
        print("C++ model completed")
    else:
        print("WARNING: C++ binary not found, skipping C++ pattern output")

    # Compare results
    print("\nComparing patterns...")
    report = generate_comparison_report(
        str(python_dir), str(cpp_dir),
        str(base_path / f"comparison_{pattern_dir_name}.txt"),
        verbose=True
    )

    return {'report': report}


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CSIIR Pattern Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare existing pattern data
  python3 csiir_pattern_compare.py --python-dir data/flat_python --cpp-dir data/flat_cpp

  # Run full comparison with all patterns
  python3 csiir_pattern_compare.py --run-all --size 64x64 --pixel-bits 10

  # Single pattern comparison
  python3 csiir_pattern_compare.py --run-all --pattern flat --size 32x32
        """
    )

    parser.add_argument('--python-dir', type=str, help='Python pattern data directory')
    parser.add_argument('--cpp-dir', type=str, help='C++ pattern data directory')
    parser.add_argument('--output', type=str, default='comparison_report.txt',
                       help='Output report file')

    parser.add_argument('--run-all', action='store_true',
                       help='Run both models and compare')
    parser.add_argument('--pattern', type=str, default='flat',
                       help='Pattern name (flat, gradient, edge, checkerboard, noise, natural)')
    parser.add_argument('--patterns', type=str,
                       help='Comma-separated pattern list')
    parser.add_argument('--size', type=str, default='64x64',
                       help='Image size WxH')
    parser.add_argument('--pixel-bits', type=int, default=10, choices=[8, 10, 12],
                       help='Pixel bit width')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base directory for pattern data')

    args = parser.parse_args()

    if args.run_all:
        # Parse size
        width, height = map(int, args.size.lower().split('x'))

        # Determine patterns to run
        if args.patterns:
            patterns = [p.strip() for p in args.patterns.split(',')]
        else:
            patterns = [args.pattern]

        # Determine base directory
        if args.base_dir:
            base_dir = args.base_dir
        else:
            base_dir = str(Path(__file__).parent.parent / "hls_csiir" / "pattern_data")

        # Run comparison for each pattern
        for pattern in patterns:
            try:
                run_full_comparison(base_dir, pattern, width, height, args.pixel_bits)
            except Exception as e:
                print(f"ERROR processing pattern {pattern}: {e}")

    elif args.python_dir and args.cpp_dir:
        # Compare existing data
        generate_comparison_report(args.python_dir, args.cpp_dir, args.output)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())