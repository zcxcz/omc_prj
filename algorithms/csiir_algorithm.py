"""
CSIIR (Color Space IIR) Module - Python Algorithm Prototype

Input: YUV422 format (Y ignored, U/V interleaved processing)
Output: Blending result of adaptive average filter

Algorithm Pipeline:
1. Stage1: Sobel Filter (5x5) -> Gradient -> Window size selection
2. Stage2: Directional weighted average with mask
3. Stage3: Gradient-weighted directional blending
4. Stage4: IIR blending with previous line

Author: Algorithm Team
Date: 2026-03-09
Version: v2.0
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class CSIIRConfig:
    """Configuration registers for CSIIR module"""
    # Window size thresholds (for LUT selection)
    win_size_thresh0: int = 16  # 2x2 threshold
    win_size_thresh1: int = 24  # 3x3 threshold
    win_size_thresh2: int = 32  # 4x4 threshold
    win_size_thresh3: int = 40  # 5x5 threshold

    # Sobel gradient clip for window size LUT
    reg_siir_win_size_grad_lut: np.ndarray = field(default_factory=lambda: np.array([16, 24, 32, 40]))
    reg_siir_win_size_grad_lut_exp: np.ndarray = field(default_factory=lambda: np.array([2, 2, 2, 2]))

    # Motion protection
    reg_siir_mot_protect: int = 0
    mot_sft: np.ndarray = field(default_factory=lambda: np.array([2, 2, 2, 2]))

    # IIR blending ratios for each window size (indexed by win_size/8 - 2)
    # For win_size = 16, 24, 32, 40 -> index = 0, 1, 2, 3
    reg_siir_blending_ratio: np.ndarray = field(
        default_factory=lambda: np.array([16, 32, 48, 64])  # 0-64 scale
    )


class CSIIRFilter:
    """
    CSIIR Filter Module

    Processing flow:
    1. Input: YUV422 interleaved (UYVY format), Y channel ignored
    2. U/V channels processed alternately
    3. Stage1: Sobel 5x5 -> Gradient magnitude -> Window size selection
    4. Stage2: Directional weighted average with mask
    5. Stage3: Gradient-weighted directional blending
    6. Stage4: IIR blending with previous line
    """

    def __init__(self, config: CSIIRConfig = None):
        self.config = config or CSIIRConfig()
        self._init_kernels()

    def _init_kernels(self):
        """Initialize all filter kernels"""
        # Stage1: Sobel 5x5 kernels (simplified difference)
        self.sobel_x = np.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32)

        self.sobel_y = np.array([
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1]
        ], dtype=np.float32)

        # Stage2: Average factor kernels (center)
        self.avg_factor_c_2x2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 2, 4, 2, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.avg_factor_c_3x3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.avg_factor_c_4x4 = np.array([
            [1, 1, 2, 1, 1],
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
            [1, 1, 2, 1, 1]
        ], dtype=np.float32)

        self.avg_factor_c_5x5 = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float32)

        # Direction masks (5x5)
        self.avg_factor_mask_r = np.array([
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1]
        ], dtype=np.float32)

        self.avg_factor_mask_l = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0]
        ], dtype=np.float32)

        self.avg_factor_mask_u = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.avg_factor_mask_d = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float32)

        # Stage4: Blend factor kernels
        self.blend_factor_2x2_h = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.blend_factor_2x2_v = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.blend_factor_3x3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 4, 4, 4, 0],
            [0, 4, 4, 4, 0],
            [0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.blend_factor_4x4 = np.array([
            [1, 2, 2, 2, 1],
            [1, 4, 4, 4, 2],
            [1, 4, 4, 4, 2],
            [1, 4, 4, 4, 2],
            [1, 2, 2, 2, 1]
        ], dtype=np.float32)

        self.blend_factor_5x5 = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float32)

        # Zeros kernel
        self.zeros_5x5 = np.zeros((5, 5), dtype=np.float32)

    def sobel_filter_5x5(self, window: np.ndarray) -> Tuple[float, float, float]:
        """
        Apply 5x5 Sobel filter and return gradient components

        Args:
            window: 5x5 pixel window

        Returns:
            Tuple of (grad_h, grad_v, grad_magnitude)
        """
        grad_h = np.sum(window * self.sobel_x)
        grad_v = np.sum(window * self.sobel_y)
        grad_magnitude = np.abs(grad_h) / 5.0 + np.abs(grad_v) / 5.0
        return grad_h, grad_v, grad_magnitude

    def get_window_size_clip(self, grad: float, grad_prev: float, grad_next: float) -> int:
        """
        Determine clipped window size based on gradient LUT

        Uses max of 3 consecutive gradients for stability

        Returns:
            Clipped window size: 16, 24, 32, or 40
        """
        # Use max of adjacent gradients
        max_grad = max(grad_prev, grad, grad_next)

        # LUT lookup based on reference:
        # reg_siir_win_size_clip_y = [15, 23, 31, 39]
        # reg_siir_win_size_clip_sft = [2, 2, 2, 2]
        # win_size_grad = LUT(Max(grad), clip_y, clip_sft)
        grad_lut_y = self.config.reg_siir_win_size_grad_lut  # [16, 24, 32, 40]

        # Compute cumulative thresholds (shifted)
        # Based on hardware LUT behavior: win_size = clip_y + (grad >> clip_sft)
        # Simplified: use gradient thresholds to select window size
        thresholds = grad_lut_y.copy()

        # Window size from gradient (LUT mapping)
        if max_grad < thresholds[0]:
            win_size_grad = 16
        elif max_grad < thresholds[1]:
            win_size_grad = 24
        elif max_grad < thresholds[2]:
            win_size_grad = 32
        else:
            win_size_grad = 40

        # Clip to valid range
        win_size = int(np.clip(win_size_grad, 16, 40))

        return win_size

    def get_avg_factor_pair(self, win_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get avg0 and avg1 factor matrices based on window size

        Returns:
            Tuple of (avg0_factor_c, avg1_factor_c)
        """
        t0 = self.config.win_size_thresh0
        t1 = self.config.win_size_thresh1
        t2 = self.config.win_size_thresh2
        t3 = self.config.win_size_thresh3

        if win_size < t0:
            return self.zeros_5x5.copy(), self.avg_factor_c_2x2.copy()
        elif win_size < t1:
            return self.avg_factor_c_2x2.copy(), self.avg_factor_c_3x3.copy()
        elif win_size < t2:
            return self.avg_factor_c_3x3.copy(), self.avg_factor_c_4x4.copy()
        elif win_size < t3:
            return self.avg_factor_c_4x4.copy(), self.avg_factor_c_5x5.copy()
        else:
            return self.avg_factor_c_5x5.copy(), self.zeros_5x5.copy()

    def get_blend_factor_pair(self, win_size: int, grad_h: float, grad_v: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get blend0 and blend1 factor matrices based on window size

        Returns:
            Tuple of (blend0_factor, blend1_factor)
        """
        t0 = self.config.win_size_thresh0
        t1 = self.config.win_size_thresh1
        t2 = self.config.win_size_thresh2
        t3 = self.config.win_size_thresh3

        # Select 2x2 orientation based on gradient direction
        if np.abs(grad_h) >= np.abs(grad_v):
            blend_2x2 = self.blend_factor_2x2_h
        else:
            blend_2x2 = self.blend_factor_2x2_v

        if win_size < t0:
            return self.zeros_5x5.copy(), blend_2x2.copy()
        elif win_size < t1:
            return blend_2x2.copy(), self.blend_factor_3x3.copy()
        elif win_size < t2:
            return self.blend_factor_3x3.copy(), self.blend_factor_4x4.copy()
        elif win_size < t3:
            return self.blend_factor_4x4.copy(), self.blend_factor_5x5.copy()
        else:
            return self.blend_factor_5x5.copy(), self.zeros_5x5.copy()

    def compute_directional_averages(self, window: np.ndarray,
                                      avg_factor_c: np.ndarray) -> Dict[str, float]:
        """
        Compute 5-directional averages using mask matrices

        Args:
            window: 5x5 pixel window
            avg_factor_c: Center factor matrix

        Returns:
            Dictionary of directional averages (c, u, d, l, r)
        """
        # Apply direction masks
        avg_factor_u = avg_factor_c * self.avg_factor_mask_u
        avg_factor_d = avg_factor_c * self.avg_factor_mask_d
        avg_factor_l = avg_factor_c * self.avg_factor_mask_l
        avg_factor_r = avg_factor_c * self.avg_factor_mask_r

        def weighted_avg(win, factor):
            weight_sum = np.sum(factor)
            if weight_sum == 0:
                return 0.0
            return np.sum(win * factor) / weight_sum

        return {
            'c': weighted_avg(window, avg_factor_c),
            'u': weighted_avg(window, avg_factor_u),
            'd': weighted_avg(window, avg_factor_d),
            'l': weighted_avg(window, avg_factor_l),
            'r': weighted_avg(window, avg_factor_r)
        }

    def get_directional_gradients(self, i: int, j: int, grad_map: np.ndarray) -> Dict[str, float]:
        """
        Get 5-directional gradients from pre-computed gradient map

        Based on reference:
        - grad_u(i, j) = grad(i, j-1) for j>0, else grad(i, j)
        - grad_d(i, j) = grad(i, j+1) for j<height-1, else grad(i, j)
        - grad_l(i, j) = grad(i-1, j) for i>0, else grad(i, j)
        - grad_r(i, j) = grad(i+1, j) for i<width-1, else grad(i, j)
        - grad_c(i, j) = grad(i, j)

        Args:
            i: Column index (x coordinate)
            j: Row index (y coordinate)
            grad_map: Pre-computed gradient magnitude map

        Returns:
            Dictionary of directional gradients (before invSort)
        """
        height, width = grad_map.shape

        # Get gradients from adjacent pixels
        # grad_u: above (j-1), grad_d: below (j+1)
        # grad_l: left (i-1), grad_r: right (i+1)
        grad_c = grad_map[j, i]

        if j == 0:
            grad_u = grad_c
        else:
            grad_u = grad_map[j - 1, i]

        if j == height - 1:
            grad_d = grad_c
        else:
            grad_d = grad_map[j + 1, i]

        if i == 0:
            grad_l = grad_c
        else:
            grad_l = grad_map[j, i - 1]

        if i == width - 1:
            grad_r = grad_c
        else:
            grad_r = grad_map[j, i + 1]

        return {
            'u': grad_u,
            'd': grad_d,
            'l': grad_l,
            'r': grad_r,
            'c': grad_c
        }

    def inv_sort(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Inverse sort - sort gradients in descending order

        Larger gradient gets larger weight.
        After sorting, the values are used as weights for directional blending.

        Args:
            values: Dictionary of directional gradients

        Returns:
            Dictionary with same keys but sorted values (descending order assigned)
        """
        # Extract values and sort in descending order
        items = list(values.items())
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        # Create result dictionary maintaining original keys
        # The values are sorted and re-assigned based on descending order
        sorted_values = [v for _, v in sorted_items]
        original_keys = [k for k, _ in items]

        # Return dictionary with sorted values assigned to original keys
        # This matches hardware behavior where invSort outputs sorted values
        return {k: v for k, v in zip(original_keys, sorted_values)}

    def gradient_weighted_blend(self, avg_values: Dict[str, float],
                                  gradients: Dict[str, float]) -> float:
        """
        Compute gradient-weighted directional average (Stage 3)

        Args:
            avg_values: Directional averages
            gradients: Directional gradients

        Returns:
            Blended average value
        """
        grad_sorted = self.inv_sort(gradients)

        grad_sum = (grad_sorted['u'] + grad_sorted['d'] +
                   grad_sorted['l'] + grad_sorted['r'] + grad_sorted['c'])

        if grad_sum < 1e-6:
            # Avoid division by zero - simple average
            return (avg_values['u'] + avg_values['d'] +
                   avg_values['l'] + avg_values['r'] + avg_values['c']) / 5.0
        else:
            # Gradient-weighted average
            blended = (avg_values['u'] * grad_sorted['u'] +
                      avg_values['d'] * grad_sorted['d'] +
                      avg_values['l'] * grad_sorted['l'] +
                      avg_values['r'] * grad_sorted['r'] +
                      avg_values['c'] * grad_sorted['c']) / grad_sum
            return blended

    def iir_blend(self, dir_avg: float, prev_u: float, win_size: int) -> float:
        """
        IIR blending with previous line (Stage 4)

        Args:
            dir_avg: Current directional average
            prev_u: Previous line's up-direction average
            win_size: Current window size

        Returns:
            IIR blended value
        """
        # Get blending ratio from config
        idx = win_size // 8 - 2  # Map 16,24,32,40 -> 0,1,2,3
        idx = np.clip(idx, 0, 3)
        ratio = self.config.reg_siir_blending_ratio[idx]

        # IIR blend: ratio * current + (64 - ratio) * previous
        return (ratio * dir_avg + (64 - ratio) * prev_u) / 64.0

    def apply_blend_factor(self, iir_avg: float, blend_factor: np.ndarray,
                           window: np.ndarray) -> np.ndarray:
        """
        Apply blend factor to generate blended 5x5 window

        Based on reference:
        blend_uv_5x5 = iir_avg * blend_factor + (4 - blend_factor) * src_uv_5x5

        Args:
            iir_avg: IIR blended average
            blend_factor: Blend factor matrix
            window: 5x5 source window

        Returns:
            Blended 5x5 output window
        """
        # blend_uv = iir_avg * factor + src * (4 - factor)
        result = iir_avg * blend_factor + (4 - blend_factor) * window
        return result

    def compute_final_blend(self, blend0_uv: np.ndarray, blend1_uv: np.ndarray,
                            win_size: int) -> np.ndarray:
        """
        Compute final blend of blend0 and blend1 outputs

        Based on reference:
        win_size_clip_remain_8 = win_size_clip - (win_size_clip >> 3)
        blend_uv_5x5 = blend0_uv_5x5 * win_size_clip_remain_8 + blend1_uv_5x5 * (8 - win_size_clip_remain_8)

        Args:
            blend0_uv: Blend0 5x5 output
            blend1_uv: Blend1 5x5 output
            win_size: Window size clip value

        Returns:
            Final blended 5x5 output
        """
        # win_size_clip_remain_8 = win_size - (win_size >> 3)
        win_size_remain_8 = win_size - (win_size >> 3)
        # Final blend with weighting
        blend_uv = blend0_uv * win_size_remain_8 + blend1_uv * (8 - win_size_remain_8)
        return blend_uv

    def process_pixel(self, window_5x5: np.ndarray,
                       i: int, j: int, grad_map: np.ndarray) -> Tuple[float, Dict]:
        """
        Process a single pixel through all stages

        Args:
            window_5x5: 5x5 pixel window centered on current pixel
            i: Column index (x coordinate)
            j: Row index (y coordinate)
            grad_map: Pre-computed gradient magnitude map

        Returns:
            Tuple of (output_value, debug_info)
        """
        # Stage 1: Sobel filter
        grad_h, grad_v, grad = self.sobel_filter_5x5(window_5x5)

        # Get 3 consecutive gradients for window size (prev, current, next column)
        width = grad_map.shape[1]
        grad_prev = grad_map[j, max(0, i-1)] if i > 0 else grad
        grad_next = grad_map[j, min(width-1, i+1)] if i < width-1 else grad
        win_size = self.get_window_size_clip(grad, grad_prev, grad_next)

        # Stage 2: Directional averages
        avg0_factor_c, avg1_factor_c = self.get_avg_factor_pair(win_size)

        avg0_values = self.compute_directional_averages(window_5x5, avg0_factor_c)
        avg1_values = self.compute_directional_averages(window_5x5, avg1_factor_c)

        # Stage 3: Gradient-weighted blending using adjacent pixel gradients
        gradients = self.get_directional_gradients(i, j, grad_map)

        blend0_dir_avg = self.gradient_weighted_blend(avg0_values, gradients)
        blend1_dir_avg = self.gradient_weighted_blend(avg1_values, gradients)

        # Stage 4: IIR blending
        blend0_iir_avg = self.iir_blend(blend0_dir_avg, avg0_values['u'], win_size)
        blend1_iir_avg = self.iir_blend(blend1_dir_avg, avg1_values['u'], win_size)

        # Apply blend factors
        blend0_factor, blend1_factor = self.get_blend_factor_pair(win_size, grad_h, grad_v)

        blend0_uv = self.apply_blend_factor(blend0_iir_avg, blend0_factor, window_5x5)
        blend1_uv = self.apply_blend_factor(blend1_iir_avg, blend1_factor, window_5x5)

        # Final blend with win_size weighting
        blend_uv = self.compute_final_blend(blend0_uv, blend1_uv, win_size)
        output = blend_uv[2, 2]  # Center pixel

        debug_info = {
            'grad': grad,
            'win_size': win_size,
            'blend0_dir_avg': blend0_dir_avg,
            'blend1_dir_avg': blend1_dir_avg
        }

        return output, debug_info

    def process_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Process a single channel (U or V) through CSIIR pipeline

        Args:
            channel: 2D array of pixel values

        Returns:
            Filtered output array
        """
        height, width = channel.shape
        output = np.zeros_like(channel, dtype=np.float32)

        # Previous line storage (for IIR)
        prev_line_u = np.zeros(width, dtype=np.float32)

        # Pad image for border handling (reflect padding)
        padded = np.pad(channel.astype(np.float32), ((2, 2), (2, 2)), mode='reflect')

        # Pre-compute gradient map for entire image
        grad_map = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                window = padded[y:y+5, x:x+5]
                _, _, grad = self.sobel_filter_5x5(window)
                grad_map[y, x] = grad

        for y in range(height):
            curr_line_u = np.zeros(width, dtype=np.float32)

            for x in range(width):
                # Extract 5x5 window
                window = padded[y:y+5, x:x+5]

                # Process pixel with pre-computed gradient map
                out_val, debug = self.process_pixel(
                    window, x, y, grad_map, prev_line_u[x]
                )

                output[y, x] = out_val
                curr_line_u[x] = debug['blend0_dir_avg']  # Use as next line's prev_u

            # Update previous line for IIR
            prev_line_u = curr_line_u.copy()

        # Clamp output to valid range
        output = np.clip(output, 0, 255)

        return output

    def process_yuv422(self, yuv_data: np.ndarray) -> np.ndarray:
        """
        Process YUV422 interleaved data

        Input format: UYVY (U Y V Y U Y V Y ...)
        Y channel is ignored, U/V processed separately

        Args:
            yuv_data: YUV422 interleaved array, shape (height, width, 2)
                      where width is in pixels, each pixel has U/V

        Returns:
            Output U/V data in same format
        """
        # Extract U and V channels (Y ignored)
        # Assumes pre-separated U/V planes (not raw UYVY interleaved)
        u_channel = yuv_data[:, :, 0]
        v_channel = yuv_data[:, :, 1]

        # Process each channel
        u_output = self.process_channel(u_channel.astype(np.float32))
        v_output = self.process_channel(v_channel.astype(np.float32))

        # Combine output
        output = np.stack([u_output, v_output], axis=-1)

        return output


def create_test_image(height: int = 64, width: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a test YUV422 image with various features"""
    # Create gradient and edge patterns
    y_channel = np.ones((height, width), dtype=np.uint8) * 128  # Y (ignored)

    # U channel: gradient pattern with edges
    u_channel = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))

    # V channel: checkerboard pattern
    v_channel = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if (i // 8 + j // 8) % 2 == 0:
                v_channel[i, j] = 200
            else:
                v_channel[i, j] = 50

    # Add some edges
    u_channel[height//4:3*height//4, width//4:3*width//4] = 255
    v_channel[height//3:2*height//3, width//3:2*width//3] = 128

    return y_channel, u_channel, v_channel


def main():
    """Main function to demonstrate CSIIR algorithm"""
    print("=" * 60)
    print("CSIIR Module - Python Algorithm Prototype v2.0")
    print("=" * 60)

    # Create configuration
    config = CSIIRConfig(
        win_size_thresh0=16,
        win_size_thresh1=24,
        win_size_thresh2=32,
        win_size_thresh3=40,
        reg_siir_blending_ratio=np.array([32, 32, 32, 32])
    )

    # Initialize filter
    csiir = CSIIRFilter(config)

    # Create test image
    print("\nGenerating test image...")
    height, width = 64, 64
    _, u, v = create_test_image(height, width)  # Y channel ignored

    # Prepare YUV422 format (U, V channels only for this demo)
    yuv_input = np.stack([u, v], axis=-1)

    print(f"Input shape: {yuv_input.shape}")
    print(f"U channel range: [{u.min()}, {u.max()}]")
    print(f"V channel range: [{v.min()}, {v.max()}]")

    # Process through CSIIR
    print("\nProcessing through CSIIR 4-stage pipeline...")
    print("  Stage1: Sobel 5x5 -> Gradient -> Window size selection")
    print("  Stage2: Directional weighted average with mask")
    print("  Stage3: Gradient-weighted directional blending")
    print("  Stage4: IIR blending with previous line")

    output = csiir.process_yuv422(yuv_input)

    print(f"\nOutput shape: {output.shape}")
    print(f"U output range: [{output[:,:,0].min():.2f}, {output[:,:,0].max():.2f}]")
    print(f"V output range: [{output[:,:,1].min():.2f}, {output[:,:,1].max():.2f}]")

    # Print sample pixels
    print("\n--- Sample pixel comparison ---")
    print(f"Input U[32, 32]: {u[32, 32]} -> Output: {output[32, 32, 0]:.2f}")
    print(f"Input V[32, 32]: {v[32, 32]} -> Output: {output[32, 32, 1]:.2f}")

    print("\n" + "=" * 60)
    print("Algorithm prototype completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()