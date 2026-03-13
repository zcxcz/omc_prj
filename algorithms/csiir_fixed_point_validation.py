"""
CSIIR Fixed-Point Validation Script

This script validates the fixed-point implementation against floating-point reference.
Measures PSNR, MSE, and maximum pixel error.

Key: Both implementations use the SAME algorithm logic, only differing in data types.

Author: HLS Team
Date: 2026-03-13
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class FixedPointConfig:
    """Fixed-point bit widths configuration"""
    pixel_bits: int = 8
    sobel_coeff_bits: int = 5
    grad_signed_bits: int = 16
    grad_bits: int = 16
    acc_bits: int = 32
    winsize_bits: int = 3
    coeff_bits: int = 8


def clamp_uint8(value: np.ndarray) -> np.ndarray:
    """Clamp values to uint8 range [0, 255]"""
    return np.clip(np.round(value), 0, 255).astype(np.uint8)


# =============================================================================
# COMMON KERNELS (used by both floating and fixed-point implementations)
# =============================================================================

# Sobel kernels (difference-based, not standard Sobel)
SOBEL_X = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1]
], dtype=np.int32)

SOBEL_Y = np.array([
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1]
], dtype=np.int32)

# Average factor kernels
AVG_FACTOR_2x2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 2, 4, 2, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

AVG_FACTOR_3x3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

AVG_FACTOR_4x4 = np.array([
    [1, 1, 2, 1, 1],
    [1, 2, 4, 2, 1],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1],
    [1, 1, 2, 1, 1]
], dtype=np.int32)

AVG_FACTOR_5x5 = np.ones((5, 5), dtype=np.int32)

# Direction masks
MASK_U = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

MASK_D = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
], dtype=np.int32)

MASK_L = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0]
], dtype=np.int32)

MASK_R = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1]
], dtype=np.int32)

# Blend factor kernels (scale factor = 4)
BLEND_2x2_H = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

BLEND_2x2_V = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

BLEND_3x3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0],
    [0, 4, 4, 4, 0],
    [0, 4, 4, 4, 0],
    [0, 0, 0, 0, 0]
], dtype=np.int32)

BLEND_4x4 = np.array([
    [4, 8, 8, 8, 4],
    [4, 16, 16, 16, 8],
    [4, 16, 16, 16, 8],
    [4, 16, 16, 16, 8],
    [4, 8, 8, 8, 4]
], dtype=np.int32)

BLEND_5x5 = np.full((5, 5), 4, dtype=np.int32)

ZEROS_5x5 = np.zeros((5, 5), dtype=np.int32)


# =============================================================================
# FLOATING-POINT IMPLEMENTATION
# =============================================================================

class FloatingPointCSIIR:
    """Floating-point reference implementation - matches algorithm exactly"""

    def sobel_filter(self, window: np.ndarray) -> Tuple[float, float, float]:
        """Stage 1: Sobel filter"""
        win = window.astype(np.float64)
        grad_h = float(np.sum(win * SOBEL_X))
        grad_v = float(np.sum(win * SOBEL_Y))
        grad = abs(grad_h) / 5.0 + abs(grad_v) / 5.0
        return grad_h, grad_v, grad

    def get_window_size(self, grad: float, grad_prev: float, grad_next: float,
                         thresh: np.ndarray) -> int:
        """Stage 2: Window size selection"""
        max_grad = max(grad_prev, grad, grad_next)
        if max_grad < thresh[0]:
            return 16
        elif max_grad < thresh[1]:
            return 24
        elif max_grad < thresh[2]:
            return 32
        else:
            return 40

    def compute_directional_avgs(self, window: np.ndarray, factor_c: np.ndarray) -> Dict[str, float]:
        """Stage 2: Compute 5-directional averages"""
        win = window.astype(np.float64)
        fc = factor_c.astype(np.float64)

        factor_u = fc * MASK_U
        factor_d = fc * MASK_D
        factor_l = fc * MASK_L
        factor_r = fc * MASK_R

        def weighted_avg(w, f):
            s = float(np.sum(f))
            if s == 0:
                return 0.0
            return float(np.sum(w * f)) / s

        return {
            'c': weighted_avg(win, fc),
            'u': weighted_avg(win, factor_u),
            'd': weighted_avg(win, factor_d),
            'l': weighted_avg(win, factor_l),
            'r': weighted_avg(win, factor_r)
        }

    def get_avg_factors(self, win_size: int, thresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get avg0 and avg1 factor matrices"""
        if win_size < thresh[0]:
            return ZEROS_5x5.copy(), AVG_FACTOR_2x2.copy()
        elif win_size < thresh[1]:
            return AVG_FACTOR_2x2.copy(), AVG_FACTOR_3x3.copy()
        elif win_size < thresh[2]:
            return AVG_FACTOR_3x3.copy(), AVG_FACTOR_4x4.copy()
        elif win_size < thresh[3]:
            return AVG_FACTOR_4x4.copy(), AVG_FACTOR_5x5.copy()
        else:
            return AVG_FACTOR_5x5.copy(), ZEROS_5x5.copy()

    def get_blend_factors(self, win_size: int, grad_h: float, grad_v: float,
                          thresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get blend0 and blend1 factor matrices"""
        blend_2x2 = BLEND_2x2_H if abs(grad_h) >= abs(grad_v) else BLEND_2x2_V

        if win_size < thresh[0]:
            return ZEROS_5x5.copy(), blend_2x2.copy()
        elif win_size < thresh[1]:
            return blend_2x2.copy(), BLEND_3x3.copy()
        elif win_size < thresh[2]:
            return BLEND_3x3.copy(), BLEND_4x4.copy()
        elif win_size < thresh[3]:
            return BLEND_4x4.copy(), BLEND_5x5.copy()
        else:
            return BLEND_5x5.copy(), ZEROS_5x5.copy()

    def get_directional_grads(self, i: int, j: int, grad_map: np.ndarray) -> Dict[str, float]:
        """Get directional gradients from gradient map"""
        height, width = grad_map.shape
        grad_c = float(grad_map[j, i])
        grad_u = grad_c if j == 0 else float(grad_map[j - 1, i])
        grad_d = grad_c if j == height - 1 else float(grad_map[j + 1, i])
        grad_l = grad_c if i == 0 else float(grad_map[j, i - 1])
        grad_r = grad_c if i == width - 1 else float(grad_map[j, i + 1])
        return {'u': grad_u, 'd': grad_d, 'l': grad_l, 'r': grad_r, 'c': grad_c}

    def inv_sort(self, grads: Dict[str, float]) -> Dict[str, float]:
        """Inverse sort gradients for weighting"""
        items = list(grads.items())
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        sorted_values = [v for _, v in sorted_items]
        original_keys = [k for k, _ in items]
        return {k: v for k, v in zip(original_keys, sorted_values)}

    def gradient_weighted_avg(self, avgs: Dict[str, float], grads: Dict[str, float]) -> float:
        """Stage 3: Gradient-weighted average"""
        grad_sorted = self.inv_sort(grads)
        grad_sum = sum(grad_sorted.values())

        if grad_sum < 1e-6:
            return sum(avgs.values()) / 5.0

        weighted = sum(avgs[k] * grad_sorted[k] for k in ['u', 'd', 'l', 'r', 'c'])
        return weighted / grad_sum

    def iir_blend(self, dir_avg: float, prev_u: float, win_size: int,
                  blend_ratio: np.ndarray) -> float:
        """Stage 4: IIR blending"""
        idx = max(0, min(3, win_size // 8 - 2))
        ratio = float(blend_ratio[idx])
        return (ratio * dir_avg + (64.0 - ratio) * prev_u) / 64.0

    def apply_blend(self, iir_avg: float, factor: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Apply blend factor to window: blend = (iir * factor + src * (16 - factor)) / 16"""
        # Factor is scaled by 4: original range [0,4] -> scaled [0,16]
        # Formula: blend = (iir * factor + src * (16 - factor)) / 16
        win = window.astype(np.float64)
        f = factor.astype(np.float64)
        return (iir_avg * f + win * (16.0 - f)) / 16.0

    def final_blend(self, blend0: np.ndarray, blend1: np.ndarray, win_size: int) -> np.ndarray:
        """Final blend - use index-based weighting"""
        # idx = (win_size - 16) // 8 = 0, 1, 2, 3 for win_size = 16, 24, 32, 40
        # blend0_weight = 2*idx + 1 = 1, 3, 5, 7
        # blend1_weight = 7 - 2*idx = 7, 5, 3, 1
        # Sum = 8 for efficient division
        idx = (win_size - 16) // 8
        blend0_weight = 2 * idx + 1
        blend1_weight = 7 - 2 * idx
        return (blend0 * blend0_weight + blend1 * blend1_weight) / 8.0

    def process_channel(self, channel: np.ndarray, thresh: np.ndarray = None,
                        blend_ratio: np.ndarray = None) -> np.ndarray:
        """Process single channel"""
        if thresh is None:
            thresh = np.array([16.0, 24.0, 32.0, 40.0])
        if blend_ratio is None:
            blend_ratio = np.array([32.0, 32.0, 32.0, 32.0])

        height, width = channel.shape
        output = np.zeros((height, width), dtype=np.float64)
        padded = np.pad(channel.astype(np.float64), ((2, 2), (2, 2)), mode='reflect')

        # Pre-compute gradient map
        grad_map = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                _, _, grad = self.sobel_filter(padded[y:y+5, x:x+5])
                grad_map[y, x] = grad

        # Process each pixel
        for y in range(height):
            for x in range(width):
                window = padded[y:y+5, x:x+5]
                grad_h, grad_v, grad = self.sobel_filter(window)

                grad_prev = grad_map[y, max(0, x-1)] if x > 0 else grad
                grad_next = grad_map[y, min(width-1, x+1)] if x < width-1 else grad
                win_size = self.get_window_size(grad, grad_prev, grad_next, thresh)

                avg0_factor, avg1_factor = self.get_avg_factors(win_size, thresh)
                avg0 = self.compute_directional_avgs(window, avg0_factor)
                avg1 = self.compute_directional_avgs(window, avg1_factor)

                grads = self.get_directional_grads(x, y, grad_map)
                blend0_avg = self.gradient_weighted_avg(avg0, grads)
                blend1_avg = self.gradient_weighted_avg(avg1, grads)

                blend0_iir = self.iir_blend(blend0_avg, avg0['u'], win_size, blend_ratio)
                blend1_iir = self.iir_blend(blend1_avg, avg1['u'], win_size, blend_ratio)

                blend0_factor, blend1_factor = self.get_blend_factors(win_size, grad_h, grad_v, thresh)
                blend0_uv = self.apply_blend(blend0_iir, blend0_factor, window)
                blend1_uv = self.apply_blend(blend1_iir, blend1_factor, window)

                blend_uv = self.final_blend(blend0_uv, blend1_uv, win_size)
                output[y, x] = blend_uv[2, 2]

        return output


# =============================================================================
# FIXED-POINT IMPLEMENTATION
# =============================================================================

class FixedPointCSIIR:
    """Fixed-point implementation simulating HLS behavior"""

    def __init__(self, config: FixedPointConfig = None):
        self.config = config or FixedPointConfig()

    def sobel_filter(self, window: np.ndarray) -> Tuple[int, int, int]:
        """Stage 1: Sobel filter with fixed-point arithmetic"""
        win = window.astype(np.int32)

        # Gx: ap_int<16> range [-255*5, 255*5] = [-1275, 1275]
        gx = int(np.sum(win * SOBEL_X))
        gx = max(-32768, min(32767, gx))  # Simulate ap_int<16>

        # Gy: ap_int<16>
        gy = int(np.sum(win * SOBEL_Y))
        gy = max(-32768, min(32767, gy))

        # Gradient: ap_uint<16> = |Gx|/5 + |Gy|/5
        grad = (abs(gx) + 2) // 5 + (abs(gy) + 2) // 5  # Rounded division
        grad = max(0, min(65535, grad))  # Simulate ap_uint<16>

        return gx, gy, grad

    def get_window_size(self, grad: int, grad_prev: int, grad_next: int,
                         thresh: np.ndarray) -> int:
        """Stage 2: Window size selection"""
        max_grad = max(grad_prev, grad, grad_next)
        if max_grad < thresh[0]:
            return 16
        elif max_grad < thresh[1]:
            return 24
        elif max_grad < thresh[2]:
            return 32
        else:
            return 40

    def compute_directional_avgs(self, window: np.ndarray, factor_c: np.ndarray) -> Dict[str, int]:
        """Stage 2: Compute 5-directional averages with fixed-point"""
        win = window.astype(np.int32)

        factor_u = factor_c * MASK_U
        factor_d = factor_c * MASK_D
        factor_l = factor_c * MASK_L
        factor_r = factor_c * MASK_R

        def weighted_avg(w, f):
            w_sum = int(np.sum(f))
            if w_sum == 0:
                return 0
            # Use 32-bit accumulator
            weighted = int(np.sum(w * f))
            weighted = max(0, min(0xFFFFFFFF, weighted))
            # Division with rounding
            result = (weighted + w_sum // 2) // w_sum
            return max(0, min(255, result))

        return {
            'c': weighted_avg(win, factor_c),
            'u': weighted_avg(win, factor_u),
            'd': weighted_avg(win, factor_d),
            'l': weighted_avg(win, factor_l),
            'r': weighted_avg(win, factor_r)
        }

    def get_avg_factors(self, win_size: int, thresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get avg0 and avg1 factor matrices"""
        if win_size < thresh[0]:
            return ZEROS_5x5.copy(), AVG_FACTOR_2x2.copy()
        elif win_size < thresh[1]:
            return AVG_FACTOR_2x2.copy(), AVG_FACTOR_3x3.copy()
        elif win_size < thresh[2]:
            return AVG_FACTOR_3x3.copy(), AVG_FACTOR_4x4.copy()
        elif win_size < thresh[3]:
            return AVG_FACTOR_4x4.copy(), AVG_FACTOR_5x5.copy()
        else:
            return AVG_FACTOR_5x5.copy(), ZEROS_5x5.copy()

    def get_blend_factors(self, win_size: int, grad_h: int, grad_v: int,
                          thresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get blend0 and blend1 factor matrices"""
        blend_2x2 = BLEND_2x2_H if abs(grad_h) >= abs(grad_v) else BLEND_2x2_V

        if win_size < thresh[0]:
            return ZEROS_5x5.copy(), blend_2x2.copy()
        elif win_size < thresh[1]:
            return blend_2x2.copy(), BLEND_3x3.copy()
        elif win_size < thresh[2]:
            return BLEND_3x3.copy(), BLEND_4x4.copy()
        elif win_size < thresh[3]:
            return BLEND_4x4.copy(), BLEND_5x5.copy()
        else:
            return BLEND_5x5.copy(), ZEROS_5x5.copy()

    def get_directional_grads(self, i: int, j: int, grad_map: np.ndarray) -> Dict[str, int]:
        """Get directional gradients from gradient map"""
        height, width = grad_map.shape
        grad_c = int(grad_map[j, i])
        grad_u = grad_c if j == 0 else int(grad_map[j - 1, i])
        grad_d = grad_c if j == height - 1 else int(grad_map[j + 1, i])
        grad_l = grad_c if i == 0 else int(grad_map[j, i - 1])
        grad_r = grad_c if i == width - 1 else int(grad_map[j, i + 1])
        return {'u': grad_u, 'd': grad_d, 'l': grad_l, 'r': grad_r, 'c': grad_c}

    def inv_sort(self, grads: Dict[str, int]) -> Dict[str, int]:
        """Inverse sort gradients for weighting"""
        items = list(grads.items())
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        sorted_values = [v for _, v in sorted_items]
        original_keys = [k for k, _ in items]
        return {k: v for k, v in zip(original_keys, sorted_values)}

    def gradient_weighted_avg(self, avgs: Dict[str, int], grads: Dict[str, int]) -> int:
        """Stage 3: Gradient-weighted average with fixed-point"""
        grad_sorted = self.inv_sort(grads)
        grad_sum = sum(grad_sorted.values())

        if grad_sum == 0:
            return sum(avgs.values()) // 5

        # Use 32-bit accumulator
        weighted = sum(avgs[k] * grad_sorted[k] for k in ['u', 'd', 'l', 'r', 'c'])
        weighted = max(0, min(0xFFFFFFFF, weighted))
        grad_sum = max(1, min(0xFFFFFFFF, grad_sum))

        result = (weighted + grad_sum // 2) // grad_sum
        return max(0, min(255, result))

    def iir_blend(self, dir_avg: int, prev_u: int, win_size: int,
                  blend_ratio: np.ndarray) -> int:
        """Stage 4: IIR blending with fixed-point"""
        idx = max(0, min(3, win_size // 8 - 2))
        ratio = int(blend_ratio[idx])

        temp = ratio * dir_avg + (64 - ratio) * prev_u
        temp = max(0, min(0xFFFFFFFF, temp))

        result = (temp + 32) // 64
        return max(0, min(255, result))

    def apply_blend(self, iir_avg: int, factor: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Apply blend factor: (iir * factor + src * (16 - factor) + 8) >> 4"""
        # Factor is scaled by 4: original range [0,4] -> scaled [0,16]
        win = window.astype(np.int32)
        result = (iir_avg * factor + win * (16 - factor) + 8) // 16
        return np.clip(result, 0, 255).astype(np.int32)

    def final_blend(self, blend0: np.ndarray, blend1: np.ndarray, win_size: int) -> np.ndarray:
        """Final blend with fixed-point - use index-based weighting"""
        # idx = (win_size - 16) // 8 = 0, 1, 2, 3 for win_size = 16, 24, 32, 40
        # blend0_weight = 2*idx + 1 = 1, 3, 5, 7
        # blend1_weight = 7 - 2*idx = 7, 5, 3, 1
        idx = (win_size - 16) // 8
        blend0_weight = 2 * idx + 1
        blend1_weight = 7 - 2 * idx
        result = (blend0 * blend0_weight + blend1 * blend1_weight + 4) // 8
        return np.clip(result, 0, 255).astype(np.int32)

    def process_channel(self, channel: np.ndarray, thresh: np.ndarray = None,
                        blend_ratio: np.ndarray = None) -> np.ndarray:
        """Process single channel with fixed-point"""
        if thresh is None:
            thresh = np.array([16, 24, 32, 40], dtype=np.int32)
        if blend_ratio is None:
            blend_ratio = np.array([32, 32, 32, 32], dtype=np.int32)

        height, width = channel.shape
        output = np.zeros((height, width), dtype=np.uint8)
        padded = np.pad(channel.astype(np.int32), ((2, 2), (2, 2)), mode='reflect')

        # Pre-compute gradient map
        grad_map = np.zeros((height, width), dtype=np.uint16)
        for y in range(height):
            for x in range(width):
                _, _, grad = self.sobel_filter(padded[y:y+5, x:x+5])
                grad_map[y, x] = grad

        # Process each pixel
        for y in range(height):
            for x in range(width):
                window = padded[y:y+5, x:x+5]
                grad_h, grad_v, grad = self.sobel_filter(window)

                grad_prev = int(grad_map[y, max(0, x-1)]) if x > 0 else grad
                grad_next = int(grad_map[y, min(width-1, x+1)]) if x < width-1 else grad
                win_size = self.get_window_size(grad, grad_prev, grad_next, thresh)

                avg0_factor, avg1_factor = self.get_avg_factors(win_size, thresh)
                avg0 = self.compute_directional_avgs(window, avg0_factor)
                avg1 = self.compute_directional_avgs(window, avg1_factor)

                grads = self.get_directional_grads(x, y, grad_map)
                blend0_avg = self.gradient_weighted_avg(avg0, grads)
                blend1_avg = self.gradient_weighted_avg(avg1, grads)

                blend0_iir = self.iir_blend(blend0_avg, avg0['u'], win_size, blend_ratio)
                blend1_iir = self.iir_blend(blend1_avg, avg1['u'], win_size, blend_ratio)

                blend0_factor, blend1_factor = self.get_blend_factors(win_size, grad_h, grad_v, thresh)
                blend0_uv = self.apply_blend(blend0_iir, blend0_factor, window)
                blend1_uv = self.apply_blend(blend1_iir, blend1_factor, window)

                blend_uv = self.final_blend(blend0_uv, blend1_uv, win_size)
                output[y, x] = int(blend_uv[2, 2])

        return output


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Mean Squared Error"""
    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


def compute_max_error(original: np.ndarray, processed: np.ndarray) -> int:
    """Compute maximum absolute error"""
    return int(np.max(np.abs(original.astype(np.int32) - processed.astype(np.int32))))


def create_test_patterns(height: int = 64, width: int = 64) -> Dict[str, np.ndarray]:
    """Create various test patterns"""
    patterns = {}

    patterns['flat'] = np.full((height, width), 128, dtype=np.uint8)
    patterns['gradient'] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))

    checker = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            checker[i, j] = 200 if (i // 8 + j // 8) % 2 == 0 else 50
    patterns['checkerboard'] = checker

    edge = np.zeros((height, width), dtype=np.uint8)
    edge[height//4:3*height//4, width//4:3*width//4] = 255
    patterns['edge'] = edge

    np.random.seed(42)
    patterns['noise'] = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    natural = np.zeros((height, width), dtype=np.int32)
    natural[:, :] = np.linspace(0, 128, width, dtype=np.int32)
    natural[height//3:2*height//3, width//3:2*width//3] = 200
    natural += np.random.randint(-20, 20, (height, width))
    patterns['natural'] = clamp_uint8(natural)

    return patterns


def run_validation(height: int = 64, width: int = 64, verbose: bool = True):
    """Run full validation suite"""
    print("=" * 70)
    print("CSIIR Fixed-Point Validation")
    print("=" * 70)
    print(f"\nImage size: {height} x {width}")
    print("Fixed-point config:")
    print("  - Pixel: uint8")
    print("  - Gradient: int16/uint16")
    print("  - Accumulator: uint32")
    print()

    fp_csiir = FloatingPointCSIIR()
    fx_csiir = FixedPointCSIIR()

    thresh = np.array([16, 24, 32, 40])
    blend_ratio = np.array([32, 32, 32, 32])

    patterns = create_test_patterns(height, width)
    results = {}

    for name, pattern in patterns.items():
        if verbose:
            print(f"\n--- Pattern: {name} ---")

        # Floating-point processing
        fp_output = fp_csiir.process_channel(pattern, thresh.astype(np.float64),
                                              blend_ratio.astype(np.float64))
        fp_output_clamped = clamp_uint8(fp_output)

        # Fixed-point processing
        fx_output = fx_csiir.process_channel(pattern, thresh, blend_ratio)

        # Compute metrics
        psnr = compute_psnr(fp_output_clamped, fx_output)
        mse = compute_mse(fp_output_clamped, fx_output)
        max_err = compute_max_error(fp_output_clamped, fx_output)

        results[name] = {'psnr': psnr, 'mse': mse, 'max_error': max_err}

        if verbose:
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  MSE:  {mse:.4f}")
            print(f"  Max Error: {max_err}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_psnr = [r['psnr'] for r in results.values() if r['psnr'] != float('inf')]
    all_mse = [r['mse'] for r in results.values()]
    all_max_err = [r['max_error'] for r in results.values()]

    print(f"\nMin PSNR: {min(all_psnr):.2f} dB")
    print(f"Max PSNR: {max(all_psnr):.2f} dB")
    print(f"Avg PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"\nMax MSE: {max(all_mse):.4f}")
    print(f"Max Error: {max(all_max_err)} pixels")

    # Pass/Fail criteria
    print("\n" + "-" * 70)
    print("PASS/FAIL CRITERIA")
    print("-" * 70)

    passed = True
    criteria = [
        ("PSNR >= 40 dB", min(all_psnr) >= 40),
        ("Max Error <= 2", max(all_max_err) <= 2),
        ("Max MSE <= 1.0", max(all_mse) <= 1.0)
    ]

    for criterion, status in criteria:
        print(f"  {criterion}: {'PASS' if status else 'FAIL'}")
        if not status:
            passed = False

    print("\n" + "=" * 70)
    if passed:
        print("OVERALL: PASS - Fixed-point meets accuracy requirements")
    else:
        print("OVERALL: FAIL - Fixed-point needs optimization")
    print("=" * 70)

    return results, passed


if __name__ == "__main__":
    results, passed = run_validation(height=64, width=64)