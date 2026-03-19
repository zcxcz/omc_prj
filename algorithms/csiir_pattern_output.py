"""
CSIIR Pattern Output Model

Wrapper for Python CSIIR implementation that outputs intermediate data
at each pipeline stage for comparison with C++ HLS implementation.

Author: HLS Team
Date: 2026-03-18
Version: 1.0
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from csiir_pattern_format import PatternData, PatternConfig
from csiir_fixed_point_validation import (
    FixedPointCSIIR, FixedPointConfig,
    SOBEL_X, SOBEL_Y,
    AVG_FACTOR_2x2, AVG_FACTOR_3x3, AVG_FACTOR_4x4, AVG_FACTOR_5x5,
    MASK_U, MASK_D, MASK_L, MASK_R,
    BLEND_2x2_H, BLEND_2x2_V, BLEND_3x3, BLEND_4x4, BLEND_5x5,
    ZEROS_5x5
)


class PatternOutputCSIIR:
    """
    CSIIR implementation with pattern output capability.

    Wraps FixedPointCSIIR to capture intermediate data at each stage:
    - Stage 1: Sobel filter (grad_h, grad_v, grad_magnitude)
    - Stage 2: Window selector (win_size, grad_used)
    - Stage 3: Directional filter (avg_c, avg_u, avg_d, avg_l, avg_r, blend0_avg, blend1_avg)
    - Stage 4: Blending (blend0_iir, blend1_iir, final_output)
    """

    def __init__(self, pixel_bits: int = 10):
        self.pixel_bits = pixel_bits
        self.pixel_max = (1 << pixel_bits) - 1
        config = FixedPointConfig(pixel_bits=pixel_bits)
        self.csiir = FixedPointCSIIR(config)

    def process_channel_with_pattern(
        self,
        channel: np.ndarray,
        thresh: np.ndarray = None,
        blend_ratio: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process single channel and return intermediate pattern data.

        Args:
            channel: Input channel array (height, width)
            thresh: Sobel thresholds [t0, t1, t2, t3]
            blend_ratio: IIR blend ratios for each window size

        Returns:
            Tuple of (output_channel, pattern_dict)
        """
        if thresh is None:
            scale = (1 << self.pixel_bits) // 256
            thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale
        if blend_ratio is None:
            blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

        height, width = channel.shape

        # Initialize output arrays
        output = np.zeros((height, width), dtype=np.uint16)
        padded = np.pad(channel.astype(np.int32), ((2, 2), (2, 2)), mode='reflect')

        # Stage 1: Sobel outputs
        grad_h_map = np.zeros((height, width), dtype=np.int32)
        grad_v_map = np.zeros((height, width), dtype=np.int32)
        grad_magnitude_map = np.zeros((height, width), dtype=np.uint32)

        # Stage 2: Window selector outputs
        win_size_map = np.zeros((height, width), dtype=np.uint8)
        grad_used_map = np.zeros((height, width), dtype=np.uint32)

        # Stage 3: Directional filter outputs
        avg_c_map = np.zeros((height, width), dtype=np.uint16)
        avg_u_map = np.zeros((height, width), dtype=np.uint16)
        avg_d_map = np.zeros((height, width), dtype=np.uint16)
        avg_l_map = np.zeros((height, width), dtype=np.uint16)
        avg_r_map = np.zeros((height, width), dtype=np.uint16)
        blend0_avg_map = np.zeros((height, width), dtype=np.uint16)
        blend1_avg_map = np.zeros((height, width), dtype=np.uint16)

        # Stage 4: Blending outputs
        blend0_iir_map = np.zeros((height, width), dtype=np.uint16)
        blend1_iir_map = np.zeros((height, width), dtype=np.uint16)

        # Pre-compute gradient map
        grad_map = np.zeros((height, width), dtype=np.uint32)
        for y in range(height):
            for x in range(width):
                gx, gy, grad = self.csiir.sobel_filter(padded[y:y+5, x:x+5])
                grad_map[y, x] = grad
                grad_h_map[y, x] = gx
                grad_v_map[y, x] = gy
                grad_magnitude_map[y, x] = grad

        # Process each pixel
        for y in range(height):
            for x in range(width):
                window = padded[y:y+5, x:x+5]
                gx, gy, grad = self.csiir.sobel_filter(window)

                grad_prev = int(grad_map[y, max(0, x-1)]) if x > 0 else grad
                grad_next = int(grad_map[y, min(width-1, x+1)]) if x < width-1 else grad
                win_size = self.csiir.get_window_size(grad, grad_prev, grad_next, thresh)

                # Store Stage 2 data
                win_size_map[y, x] = win_size
                grad_used_map[y, x] = max(grad_prev, grad, grad_next)

                avg0_factor, avg1_factor = self.csiir.get_avg_factors(win_size, thresh)
                avg0 = self.csiir.compute_directional_avgs(window, avg0_factor)
                avg1 = self.csiir.compute_directional_avgs(window, avg1_factor)

                # Store Stage 3 data
                avg_c_map[y, x] = avg0['c']
                avg_u_map[y, x] = avg0['u']
                avg_d_map[y, x] = avg0['d']
                avg_l_map[y, x] = avg0['l']
                avg_r_map[y, x] = avg0['r']

                grads = self.csiir.get_directional_grads(x, y, grad_map)
                blend0_avg = self.csiir.gradient_weighted_avg(avg0, grads)
                blend1_avg = self.csiir.gradient_weighted_avg(avg1, grads)

                blend0_avg_map[y, x] = blend0_avg
                blend1_avg_map[y, x] = blend1_avg

                blend0_iir = self.csiir.iir_blend(blend0_avg, avg0['u'], win_size, blend_ratio)
                blend1_iir = self.csiir.iir_blend(blend1_avg, avg1['u'], win_size, blend_ratio)

                # Store Stage 4 data
                blend0_iir_map[y, x] = blend0_iir
                blend1_iir_map[y, x] = blend1_iir

                blend0_factor, blend1_factor = self.csiir.get_blend_factors(win_size, gx, gy, thresh)
                blend0_uv = self.csiir.apply_blend(blend0_iir, blend0_factor, window)
                blend1_uv = self.csiir.apply_blend(blend1_iir, blend1_factor, window)

                blend_uv = self.csiir.final_blend(blend0_uv, blend1_uv, win_size)
                output[y, x] = int(blend_uv[2, 2])

        # Compile pattern data
        pattern = {
            # Stage 1: Sobel
            'grad_h': grad_h_map,
            'grad_v': grad_v_map,
            'grad_magnitude': grad_magnitude_map,

            # Stage 2: Window Selector
            'win_size': win_size_map,
            'grad_used': grad_used_map,

            # Stage 3: Directional Filter
            'avg_c': avg_c_map,
            'avg_u': avg_u_map,
            'avg_d': avg_d_map,
            'avg_l': avg_l_map,
            'avg_r': avg_r_map,
            'blend0_avg': blend0_avg_map,
            'blend1_avg': blend1_avg_map,

            # Stage 4: Blending
            'blend0_iir': blend0_iir_map,
            'blend1_iir': blend1_iir_map,

            # Final output
            'final_output': output
        }

        return output, pattern

    def process_yuv_with_pattern(
        self,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        thresh: np.ndarray = None,
        blend_ratio: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Process YUV channels and return pattern data for each channel.

        Returns:
            Tuple of (y_out, u_out, v_out, patterns_dict)
            patterns_dict contains pattern data for Y, U, V channels separately
        """
        y_out, y_pattern = self.process_channel_with_pattern(y, thresh, blend_ratio)
        u_out, u_pattern = self.process_channel_with_pattern(u, thresh, blend_ratio)
        v_out, v_pattern = self.process_channel_with_pattern(v, thresh, blend_ratio)

        patterns = {
            'Y': y_pattern,
            'U': u_pattern,
            'V': v_pattern
        }

        return y_out, u_out, v_out, patterns

    def save_pattern_data(
        self,
        y: np.ndarray, u: np.ndarray, v: np.ndarray,
        output_dir: str,
        pattern_name: str = "test",
        thresh: np.ndarray = None,
        blend_ratio: np.ndarray = None
    ) -> PatternData:
        """
        Process YUV data and save pattern data to files.

        Args:
            y, u, v: Input YUV channels
            output_dir: Output directory for pattern files
            pattern_name: Name for the pattern (used in subdirectory)
            thresh, blend_ratio: Processing parameters

        Returns:
            PatternData object with all intermediate data
        """
        height, width = y.shape

        # Create config
        config = PatternConfig(
            width=width,
            height=height,
            pixel_bits=self.pixel_bits,
            channels=3,
            model="python"
        )

        # Create PatternData object
        pattern_data = PatternData(config)

        # Set input
        pattern_data.set_input(y, u, v)

        # Process each channel and store intermediate data
        for ch_name, ch_input in [('Y', y), ('U', u), ('V', v)]:
            ch_out, ch_pattern = self.process_channel_with_pattern(
                ch_input, thresh, blend_ratio
            )

            # Store intermediate data for the channel
            # We'll save per-channel patterns to separate directories
            ch_dir = Path(output_dir) / ch_name
            ch_dir.mkdir(parents=True, exist_ok=True)

            # Save Stage 1
            np.savez(
                ch_dir / "stage1_sobel.npz",
                grad_h=ch_pattern['grad_h'],
                grad_v=ch_pattern['grad_v'],
                grad_magnitude=ch_pattern['grad_magnitude']
            )

            # Save Stage 2
            np.savez(
                ch_dir / "stage2_window_selector.npz",
                win_size=ch_pattern['win_size'],
                grad_used=ch_pattern['grad_used']
            )

            # Save Stage 3
            np.savez(
                ch_dir / "stage3_directional_filter.npz",
                avg_c=ch_pattern['avg_c'],
                avg_u=ch_pattern['avg_u'],
                avg_d=ch_pattern['avg_d'],
                avg_l=ch_pattern['avg_l'],
                avg_r=ch_pattern['avg_r'],
                blend0_avg=ch_pattern['blend0_avg'],
                blend1_avg=ch_pattern['blend1_avg']
            )

            # Save Stage 4
            np.savez(
                ch_dir / "stage4_blending.npz",
                blend0_iir=ch_pattern['blend0_iir'],
                blend1_iir=ch_pattern['blend1_iir'],
                final_output=ch_pattern['final_output']
            )

        # Process all channels for final output
        y_out, u_out, v_out, _ = self.process_yuv_with_pattern(y, u, v, thresh, blend_ratio)
        pattern_data.set_output(y_out, u_out, v_out)

        # Save input/output
        pattern_data.save_input(str(Path(output_dir) / "input.npz"))
        pattern_data.save_output(str(Path(output_dir) / "output.npz"))

        # Save config
        import json
        with open(Path(output_dir) / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        print(f"Pattern data saved to: {output_dir}")
        return pattern_data


def generate_and_save_patterns(
    output_base_dir: str,
    pattern_name: str = "flat",
    width: int = 64,
    height: int = 64,
    pixel_bits: int = 10
) -> PatternData:
    """
    Generate test pattern and save all intermediate data.

    Args:
        output_base_dir: Base output directory
        pattern_name: Test pattern name (flat, gradient, edge, checkerboard, noise, natural)
        width, height: Image dimensions
        pixel_bits: Pixel bit depth

    Returns:
        PatternData object
    """
    from csiir_c2c_utils import generate_test_pattern

    pixel_max = (1 << pixel_bits) - 1

    # Generate test pattern
    y, u, v = generate_test_pattern(pattern_name, height, width, pixel_max=pixel_max)

    # Create output directory
    output_dir = Path(output_base_dir) / f"{pattern_name}_{width}x{height}_{pixel_bits}bit_python"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process and save
    csiir = PatternOutputCSIIR(pixel_bits=pixel_bits)

    # Scale thresholds for pixel bit depth
    scale = (1 << pixel_bits) // 256
    thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale
    blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

    return csiir.save_pattern_data(y, u, v, str(output_dir), pattern_name, thresh, blend_ratio)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Demo/test entry point"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("CSIIR Pattern Output Model Test")
    print("=" * 70)

    # Test with flat pattern
    pixel_bits = 10
    width, height = 32, 32
    pixel_max = (1 << pixel_bits) - 1

    print(f"\nGenerating flat test pattern ({width}x{height}, {pixel_bits}-bit)...")

    # Create test data
    y = np.full((height, width), pixel_max // 2, dtype=np.uint16)
    u = np.full((height, width), pixel_max // 2, dtype=np.uint16)
    v = np.full((height, width), pixel_max // 2, dtype=np.uint16)

    # Process
    csiir = PatternOutputCSIIR(pixel_bits=pixel_bits)

    scale = (1 << pixel_bits) // 256
    thresh = np.array([16, 24, 32, 40], dtype=np.int32) * scale
    blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

    y_out, u_out, v_out, patterns = csiir.process_yuv_with_pattern(
        y, u, v, thresh, blend_ratio
    )

    print(f"\nOutput shape: {y_out.shape}")
    print(f"Y output range: [{y_out.min()}, {y_out.max()}]")

    # Print pattern data summary
    for ch_name, ch_pattern in patterns.items():
        print(f"\n{ch_name} Channel Pattern Data:")
        print(f"  Stage 1 - grad_magnitude range: [{ch_pattern['grad_magnitude'].min()}, {ch_pattern['grad_magnitude'].max()}]")
        print(f"  Stage 2 - win_size distribution: {np.bincount(ch_pattern['win_size'].flatten(), minlength=6)}")
        print(f"  Stage 3 - blend0_avg range: [{ch_pattern['blend0_avg'].min()}, {ch_pattern['blend0_avg'].max()}]")
        print(f"  Stage 4 - final_output range: [{ch_pattern['final_output'].min()}, {ch_pattern['final_output'].max()}]")

    # Save to file
    output_dir = "/tmp/test_python_pattern"
    print(f"\nSaving pattern data to {output_dir}...")
    csiir.save_pattern_data(y, u, v, output_dir, "flat", thresh, blend_ratio)

    print("\n" + "=" * 70)
    print("Pattern output model test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()