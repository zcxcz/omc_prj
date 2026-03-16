"""
CSIIR C2C (C-to-C) Verification Utilities

Binary data exchange between Python reference and C++ HLS implementation.
Supports 8-bit and 10-bit pixel formats.

Features:
- Binary file I/O with compact header
- Test pattern generation
- Golden reference generation via Python algorithm
- PSNR/MSE metrics computation
- NPZ visualization output

Author: HLS Team
Date: 2026-03-15
Version: 1.0
"""

import numpy as np
import struct
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import subprocess
import os


# =============================================================================
# Binary File Format
# =============================================================================

# Magic number for CSIIR binary files
CSIIR_MAGIC = b"CSIIR"
HEADER_SIZE = 16  # bytes


@dataclass
class BinaryHeader:
    """Binary file header structure"""
    magic: bytes = CSIIR_MAGIC
    width: int = 0      # uint16
    height: int = 0     # uint16
    pixel_bits: int = 8 # uint8 (8, 10, or 12)
    channels: int = 3   # uint8 (1=Y, 2=UV, 3=YUV)
    reserved: bytes = b'\x00' * 6

    def pack(self) -> bytes:
        """Pack header to binary"""
        return struct.pack(
            '<4s HH BB 6s',
            self.magic,
            self.width,
            self.height,
            self.pixel_bits,
            self.channels,
            self.reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'BinaryHeader':
        """Unpack header from binary"""
        magic, width, height, pixel_bits, channels, reserved = struct.unpack(
            '<4s HH BB 6s', data
        )
        return cls(magic, width, height, pixel_bits, channels, reserved)


# =============================================================================
# Binary I/O Functions
# =============================================================================

def save_binary(
    filepath: str,
    data: np.ndarray,
    pixel_bits: int = 10,
    channels: int = 3
) -> None:
    """
    Save image data to binary file.

    Args:
        filepath: Output file path
        data: Image array, shape (height, width, channels) or (height, width)
        pixel_bits: Bits per pixel (8, 10, or 12)
        channels: Number of channels (1, 2, or 3)
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
        channels = 1

    height, width, ch = data.shape
    assert ch == channels, f"Channel mismatch: {ch} != {channels}"

    # Create header
    header = BinaryHeader(
        width=width,
        height=height,
        pixel_bits=pixel_bits,
        channels=channels
    )

    # Convert to appropriate dtype
    if pixel_bits <= 8:
        data_out = data.astype(np.uint8)
    elif pixel_bits <= 16:
        data_out = data.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported pixel_bits: {pixel_bits}")

    # Write binary file
    with open(filepath, 'wb') as f:
        f.write(header.pack())
        f.write(data_out.tobytes())


def load_binary(filepath: str) -> Tuple[np.ndarray, BinaryHeader]:
    """
    Load image data from binary file.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (data array, header)
    """
    with open(filepath, 'rb') as f:
        header_data = f.read(HEADER_SIZE)
        header = BinaryHeader.unpack(header_data)

        # Read pixel data
        bytes_per_pixel = (header.pixel_bits + 7) // 8
        total_pixels = header.width * header.height * header.channels
        pixel_data = f.read(total_pixels * bytes_per_pixel)

    # Convert to numpy array
    if header.pixel_bits <= 8:
        dtype = np.uint8
    else:
        dtype = np.uint16

    data = np.frombuffer(pixel_data, dtype=dtype)
    data = data.reshape((header.height, header.width, header.channels))

    return data, header


def save_binary_input(
    filepath: str,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_bits: int = 10
) -> None:
    """
    Save YUV input to binary file (YUV444 format).

    Args:
        filepath: Output file path
        y, u, v: Y, U, V channel arrays (height, width)
        pixel_bits: Bits per pixel
    """
    height, width = y.shape
    data = np.stack([y, u, v], axis=-1)
    save_binary(filepath, data, pixel_bits=pixel_bits, channels=3)


def load_binary_output(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, BinaryHeader]:
    """
    Load YUV output from binary file.

    Returns:
        Tuple of (y, u, v, header)
    """
    data, header = load_binary(filepath)
    y = data[:, :, 0]
    u = data[:, :, 1]
    v = data[:, :, 2]
    return y, u, v, header


# =============================================================================
# Test Pattern Generation
# =============================================================================

def generate_test_pattern(
    name: str,
    height: int,
    width: int,
    pixel_max: int = 1023
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test pattern for Y, U, V channels.

    Args:
        name: Pattern name ('flat', 'gradient', 'edge', 'checkerboard', 'noise', 'natural')
        height: Image height
        width: Image width
        pixel_max: Maximum pixel value (e.g., 1023 for 10-bit, 255 for 8-bit)

    Returns:
        Tuple of (y, u, v) channel arrays
    """
    patterns = {
        'flat': _gen_flat,
        'gradient': _gen_gradient,
        'edge': _gen_edge,
        'checkerboard': _gen_checkerboard,
        'noise': _gen_noise,
        'natural': _gen_natural
    }

    if name not in patterns:
        raise ValueError(f"Unknown pattern: {name}. Available: {list(patterns.keys())}")

    return patterns[name](height, width, pixel_max)


def _gen_flat(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform flat region - tests static processing"""
    mid = pixel_max // 2
    y = np.full((height, width), mid, dtype=np.uint16)
    u = np.full((height, width), mid, dtype=np.uint16)
    v = np.full((height, width), mid, dtype=np.uint16)
    return y, u, v


def _gen_gradient(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth gradient - tests smooth transition handling"""
    y = np.tile(np.linspace(0, pixel_max, width, dtype=np.uint16), (height, 1))
    u = np.tile(np.linspace(0, pixel_max, height, dtype=np.uint16)[:, np.newaxis], (1, width))
    v = np.full((height, width), pixel_max // 2, dtype=np.uint16)
    return y, u, v


def _gen_edge(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sharp edges - tests edge preservation"""
    y = np.zeros((height, width), dtype=np.uint16)
    u = np.zeros((height, width), dtype=np.uint16)
    v = np.zeros((height, width), dtype=np.uint16)

    # Create rectangular region with high value
    h4, w4 = height // 4, width // 4
    y[h4:3*h4, w4:3*w4] = pixel_max
    u[h4:3*h4, w4:3*w4] = pixel_max * 3 // 4
    v[h4:3*h4, w4:3*w4] = pixel_max // 2

    return y, u, v


def _gen_checkerboard(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checkerboard pattern - tests high-frequency detail"""
    y = np.zeros((height, width), dtype=np.uint16)
    u = np.zeros((height, width), dtype=np.uint16)
    v = np.zeros((height, width), dtype=np.uint16)

    block_size = max(8, min(height, width) // 8)
    for i in range(height):
        for j in range(width):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                y[i, j] = pixel_max * 80 // 100
                u[i, j] = pixel_max * 60 // 100
                v[i, j] = pixel_max * 70 // 100
            else:
                y[i, j] = pixel_max * 20 // 100
                u[i, j] = pixel_max * 40 // 100
                v[i, j] = pixel_max * 30 // 100

    return y, u, v


def _gen_noise(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random noise - tests noise suppression"""
    np.random.seed(42)  # Reproducible
    y = np.random.randint(0, pixel_max + 1, (height, width), dtype=np.uint16)
    u = np.random.randint(0, pixel_max + 1, (height, width), dtype=np.uint16)
    v = np.random.randint(0, pixel_max + 1, (height, width), dtype=np.uint16)
    return y, u, v


def _gen_natural(height: int, width: int, pixel_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Natural-like image - combination of features"""
    np.random.seed(123)

    # Base gradient
    y = np.tile(np.linspace(0, pixel_max // 2, width, dtype=np.float32), (height, 1))
    u = np.tile(np.linspace(pixel_max // 4, pixel_max // 2, height, dtype=np.float32)[:, np.newaxis], (1, width))
    v = np.full((height, width), pixel_max // 2, dtype=np.float32)

    # Add rectangular feature
    h3, w3 = height // 3, width // 3
    y[h3:2*h3, w3:2*w3] = pixel_max * 0.8
    u[h3:2*h3, w3:2*w3] = pixel_max * 0.6
    v[h3:2*h3, w3:2*w3] = pixel_max * 0.4

    # Add noise
    y += np.random.normal(0, pixel_max * 0.05, (height, width))
    u += np.random.normal(0, pixel_max * 0.03, (height, width))
    v += np.random.normal(0, pixel_max * 0.03, (height, width))

    # Clip to valid range
    y = np.clip(y, 0, pixel_max).astype(np.uint16)
    u = np.clip(u, 0, pixel_max).astype(np.uint16)
    v = np.clip(v, 0, pixel_max).astype(np.uint16)

    return y, u, v


# =============================================================================
# Python Reference Implementation (Simplified for C2C)
# =============================================================================

def process_with_python(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    thresh: np.ndarray = None,
    blend_ratio: np.ndarray = None,
    pixel_bits: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process YUV channels with Python CSIIR algorithm.

    This is a simplified version matching the HLS C++ implementation behavior.

    Args:
        y, u, v: Input channel arrays
        thresh: Sobel thresholds for window size selection
        blend_ratio: IIR blending ratios
        pixel_bits: Bits per pixel

    Returns:
        Tuple of processed (y_out, u_out, v_out)
    """
    from csiir_fixed_point_validation import FixedPointCSIIR, create_test_patterns

    if thresh is None:
        # Scale thresholds for pixel bit depth
        scale = (1 << pixel_bits) // 256
        thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale

    if blend_ratio is None:
        blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

    # Use fixed-point implementation for C2C matching
    csiir = FixedPointCSIIR()

    # Process each channel independently
    y_out = csiir.process_channel(y.astype(np.uint8) if pixel_bits <= 8 else y, thresh, blend_ratio)
    u_out = csiir.process_channel(u.astype(np.uint8) if pixel_bits <= 8 else u, thresh, blend_ratio)
    v_out = csiir.process_channel(v.astype(np.uint8) if pixel_bits <= 8 else v, thresh, blend_ratio)

    return y_out, u_out, v_out


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_psnr(original: np.ndarray, processed: np.ndarray, pixel_max: int = 1023) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original image array
        processed: Processed image array
        pixel_max: Maximum pixel value

    Returns:
        PSNR in dB (inf if identical)
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((pixel_max ** 2) / mse)


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


def compute_max_error(original: np.ndarray, processed: np.ndarray) -> int:
    """Compute maximum absolute error."""
    return int(np.max(np.abs(original.astype(np.int32) - processed.astype(np.int32))))


def compare_results(
    golden_y: np.ndarray, golden_u: np.ndarray, golden_v: np.ndarray,
    output_y: np.ndarray, output_u: np.ndarray, output_v: np.ndarray,
    pixel_max: int = 1023
) -> Dict:
    """
    Compare golden reference with C++ output.

    Returns:
        Dictionary with PSNR, MSE, and max_error for each channel
    """
    results = {}

    for name, gold, out in [('Y', golden_y, output_y), ('U', golden_u, output_u), ('V', golden_v, output_v)]:
        results[name] = {
            'psnr': compute_psnr(gold, out, pixel_max),
            'mse': compute_mse(gold, out),
            'max_error': compute_max_error(gold, out),
            'mean_error': float(np.mean(np.abs(gold.astype(np.float64) - out.astype(np.float64))))
        }

    # Overall metrics
    results['overall'] = {
        'psnr': compute_psnr(
            np.stack([golden_y, golden_u, golden_v], axis=-1),
            np.stack([output_y, output_u, output_v], axis=-1),
            pixel_max
        ),
        'mse': (results['Y']['mse'] + results['U']['mse'] + results['V']['mse']) / 3,
        'max_error': max(results['Y']['max_error'], results['U']['max_error'], results['V']['max_error'])
    }

    return results


# =============================================================================
# NPZ Visualization Output
# =============================================================================

def save_npz(
    filepath: str,
    input_y: np.ndarray, input_u: np.ndarray, input_v: np.ndarray,
    output_y: np.ndarray, output_u: np.ndarray, output_v: np.ndarray,
    golden_y: np.ndarray = None, golden_u: np.ndarray = None, golden_v: np.ndarray = None,
    metadata: Dict = None
) -> None:
    """
    Save results to NPZ file for visualization.

    Args:
        filepath: Output .npz file path
        input_y/u/v: Input channels
        output_y/u/v: C++ output channels
        golden_y/u/v: Python golden reference (optional)
        metadata: Additional metadata dict (optional)
    """
    data = {
        'input_y': input_y,
        'input_u': input_u,
        'input_v': input_v,
        'output_y': output_y,
        'output_u': output_u,
        'output_v': output_v,
    }

    if golden_y is not None:
        data['golden_y'] = golden_y
        data['golden_u'] = golden_u
        data['golden_v'] = golden_v

        # Compute error maps
        data['error_y'] = np.abs(golden_y.astype(np.int32) - output_y.astype(np.int32)).astype(np.uint16)
        data['error_u'] = np.abs(golden_u.astype(np.int32) - output_u.astype(np.int32)).astype(np.uint16)
        data['error_v'] = np.abs(golden_v.astype(np.int32) - output_v.astype(np.int32)).astype(np.uint16)

    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (int, float, str)):
                data[f'meta_{key}'] = np.array([value])

    np.savez(filepath, **data)


def load_npz(filepath: str) -> Dict:
    """Load NPZ visualization file."""
    data = np.load(filepath)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


# =============================================================================
# C++ Testbench Execution
# =============================================================================

def compile_cpp_testbench(
    hls_dir: str,
    output_bin: str = "tb_csiir_c2c",
    debug_flags: List[str] = None,
    pixel_bitwidth: int = 10
) -> bool:
    """
    Compile C++ C2C testbench.

    Args:
        hls_dir: Path to hls_csiir directory
        output_bin: Output binary name
        debug_flags: List of debug flags (e.g., ['DEBUG_SOBEL=1'])
        pixel_bitwidth: Pixel bit width (8, 10, or 12)

    Returns:
        True if compilation succeeded
    """
    if debug_flags is None:
        debug_flags = []

    # Build compile command
    cmd = [
        'g++', '-std=c++11', '-O2',
        f'-I{hls_dir}/include',
        f'-DPIXEL_BITWIDTH={pixel_bitwidth}',
    ]

    # Add debug flags
    for flag in debug_flags:
        cmd.append(f'-D{flag}')

    # Add source files
    cmd.extend([
        f'{hls_dir}/src/sobel_filter.cpp',
        f'{hls_dir}/src/window_selector.cpp',
        f'{hls_dir}/src/directional_filter.cpp',
        f'{hls_dir}/src/blending.cpp',
        f'{hls_dir}/src/csiir_top.cpp',
        f'{hls_dir}/tb/tb_csiir_c2c.cpp',
        '-o', f'{hls_dir}/{output_bin}'
    ])

    print(f"Compiling: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Compilation timed out")
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        return False


def run_cpp_testbench(
    hls_dir: str,
    input_file: str,
    output_file: str,
    width: int,
    height: int,
    binary_name: str = "tb_csiir_c2c",
    debug_dir: str = None
) -> bool:
    """
    Run C++ C2C testbench.

    Args:
        hls_dir: Path to hls_csiir directory
        input_file: Input binary file path
        output_file: Output binary file path
        width: Image width
        height: Image height
        binary_name: Compiled binary name
        debug_dir: Directory for debug output (optional)

    Returns:
        True if execution succeeded
    """
    binary_path = f'{hls_dir}/{binary_name}'

    if not os.path.exists(binary_path):
        print(f"Binary not found: {binary_path}")
        return False

    cmd = [binary_path, input_file, output_file, str(width), str(height)]

    if debug_dir:
        cmd.append(debug_dir)

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Execution failed:\n{result.stderr}")
            return False
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print("Execution timed out")
        return False
    except Exception as e:
        print(f"Execution error: {e}")
        return False


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict,
    output_path: str,
    pattern_name: str,
    width: int,
    height: int,
    pixel_bits: int
) -> None:
    """
    Generate C2C validation report.

    Args:
        results: Comparison results dictionary
        output_path: Output report file path
        pattern_name: Test pattern name
        width, height: Image dimensions
        pixel_bits: Pixel bit width
    """
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CSIIR C2C VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Pattern:     {pattern_name}\n")
        f.write(f"Resolution:  {width} x {height}\n")
        f.write(f"Pixel Bits:  {pixel_bits}\n")
        f.write(f"Pixel Max:   {(1 << pixel_bits) - 1}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CHANNEL METRICS\n")
        f.write("-" * 70 + "\n")

        for ch in ['Y', 'U', 'V']:
            r = results[ch]
            psnr_str = f"{r['psnr']:.2f}" if r['psnr'] != float('inf') else "INF"
            f.write(f"\n{ch} Channel:\n")
            f.write(f"  PSNR:       {psnr_str} dB\n")
            f.write(f"  MSE:        {r['mse']:.4f}\n")
            f.write(f"  Max Error:  {r['max_error']}\n")
            f.write(f"  Mean Error: {r['mean_error']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        o = results['overall']
        psnr_str = f"{o['psnr']:.2f}" if o['psnr'] != float('inf') else "INF"
        f.write(f"  PSNR:       {psnr_str} dB\n")
        f.write(f"  MSE:        {o['mse']:.4f}\n")
        f.write(f"  Max Error:  {o['max_error']}\n")

        # Pass/Fail criteria
        f.write("\n" + "-" * 70 + "\n")
        f.write("PASS/FAIL CRITERIA\n")
        f.write("-" * 70 + "\n")

        passed = True
        criteria = [
            ("PSNR >= 40 dB", o['psnr'] >= 40 or o['psnr'] == float('inf')),
            ("Max Error <= 3", o['max_error'] <= 3),
            ("MSE <= 2.0", o['mse'] <= 2.0)
        ]

        for criterion, status in criteria:
            f.write(f"  {criterion}: {'PASS' if status else 'FAIL'}\n")
            if not status:
                passed = False

        f.write("\n" + "=" * 70 + "\n")
        if passed:
            f.write("OVERALL: PASS\n")
        else:
            f.write("OVERALL: FAIL\n")
        f.write("=" * 70 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Demo/test entry point."""
    print("=" * 60)
    print("CSIIR C2C Utilities Test")
    print("=" * 60)

    # Test binary I/O
    print("\nTesting binary I/O...")
    test_data = np.random.randint(0, 1024, (64, 64, 3), dtype=np.uint16)
    save_binary('/tmp/test_csiir.bin', test_data, pixel_bits=10, channels=3)
    loaded, header = load_binary('/tmp/test_csiir.bin')
    print(f"  Saved and loaded: {loaded.shape}, header: {header.width}x{header.height}, {header.pixel_bits}-bit")
    print(f"  Data match: {np.allclose(test_data, loaded)}")

    # Test pattern generation
    print("\nTesting pattern generation...")
    for name in ['flat', 'gradient', 'edge', 'checkerboard', 'noise', 'natural']:
        y, u, v = generate_test_pattern(name, 64, 64, pixel_max=1023)
        print(f"  {name}: Y[{y.min()},{y.max()}], U[{u.min()},{u.max()}], V[{v.min()},{v.max()}]")

    print("\n" + "=" * 60)
    print("C2C Utilities test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()