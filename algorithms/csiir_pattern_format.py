"""
CSIIR Pattern Data Format Specification

Unified intermediate data format for C++ HLS and Python model comparison.

Directory Structure:
-------------------
pattern_data/
├── config.json                    # Configuration used for processing
├── input.npz                      # Input YUV data
├── stage1_sobel.npz               # Sobel filter outputs
├── stage2_window_selector.npz     # Window selection outputs
├── stage3_directional_filter.npz  # Directional filter outputs
├── stage4_blending.npz            # Final blending outputs
└── output.npz                     # Final processed YUV output

Stage Data Format (NPZ files):
------------------------------

Stage 1 - Sobel (stage1_sobel.npz):
    - grad_h: int32 array (height, width) - Horizontal gradient
    - grad_v: int32 array (height, width) - Vertical gradient
    - grad_magnitude: uint32 array (height, width) - |Gx| + |Gy|

Stage 2 - Window Selector (stage2_window_selector.npz):
    - win_size: uint8 array (height, width) - Window size (2, 3, 4, or 5)
    - grad_used: uint32 array (height, width) - Gradient used for selection

Stage 3 - Directional Filter (stage3_directional_filter.npz):
    - avg_c: uint16 array (height, width) - Center average
    - avg_u: uint16 array (height, width) - Up average
    - avg_d: uint16 array (height, width) - Down average
    - avg_l: uint16 array (height, width) - Left average
    - avg_r: uint16 array (height, width) - Right average
    - blend0_avg: uint16 array (height, width) - Blend0 direction average
    - blend1_avg: uint16 array (height, width) - Blend1 direction average

Stage 4 - Blending (stage4_blending.npz):
    - blend0_iir: uint16 array (height, width) - Blend0 IIR output
    - blend1_iir: uint16 array (height, width) - Blend1 IIR output
    - final_output: uint16 array (height, width) - Final channel output

Config Format (config.json):
---------------------------
{
    "width": 64,
    "height": 64,
    "pixel_bits": 10,
    "channels": 3,
    "model": "python" or "cpp",
    "timestamp": "2026-03-17T00:00:00",
    "thresholds": [16, 24, 32, 40],
    "blend_ratios": [32, 32, 32, 32]
}

Author: HLS Team
Date: 2026-03-17
Version: 1.0
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


# =============================================================================
# Pattern Data Structures
# =============================================================================

@dataclass
class PatternConfig:
    """Configuration for pattern data"""
    width: int
    height: int
    pixel_bits: int = 10
    channels: int = 3
    model: str = "unknown"  # "python" or "cpp"
    timestamp: str = ""
    thresholds: List[int] = None
    blend_ratios: List[int] = None

    def __post_init__(self):
        if self.thresholds is None:
            # Scale thresholds for pixel bit depth
            scale = (1 << self.pixel_bits) // 256
            self.thresholds = [16 * scale, 24 * scale, 32 * scale, 40 * scale]
        if self.blend_ratios is None:
            self.blend_ratios = [32, 32, 32, 32]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'PatternConfig':
        return cls(**d)


class PatternData:
    """
    Container for CSIIR pattern data at all pipeline stages.

    Supports saving/loading to NPZ format for easy comparison between
    Python reference and C++ HLS implementations.
    """

    def __init__(self, config: PatternConfig):
        self.config = config

        # Stage 1: Sobel outputs
        self.grad_h: Optional[np.ndarray] = None
        self.grad_v: Optional[np.ndarray] = None
        self.grad_magnitude: Optional[np.ndarray] = None

        # Stage 2: Window selector outputs
        self.win_size: Optional[np.ndarray] = None
        self.grad_used: Optional[np.ndarray] = None

        # Stage 3: Directional filter outputs
        self.avg_c: Optional[np.ndarray] = None
        self.avg_u: Optional[np.ndarray] = None
        self.avg_d: Optional[np.ndarray] = None
        self.avg_l: Optional[np.ndarray] = None
        self.avg_r: Optional[np.ndarray] = None
        self.blend0_avg: Optional[np.ndarray] = None
        self.blend1_avg: Optional[np.ndarray] = None

        # Stage 4: Blending outputs
        self.blend0_iir: Optional[np.ndarray] = None
        self.blend1_iir: Optional[np.ndarray] = None
        self.final_output: Optional[np.ndarray] = None

        # Input/Output
        self.input_data: Optional[np.ndarray] = None  # (height, width, 3) YUV
        self.output_data: Optional[np.ndarray] = None  # (height, width, 3) YUV

    # -------------------------------------------------------------------------
    # Stage 1: Sobel
    # -------------------------------------------------------------------------

    def set_stage1_sobel(self, grad_h: np.ndarray, grad_v: np.ndarray,
                         grad_magnitude: np.ndarray):
        """Set Stage 1 Sobel filter outputs"""
        self.grad_h = grad_h.astype(np.int32)
        self.grad_v = grad_v.astype(np.int32)
        self.grad_magnitude = grad_magnitude.astype(np.uint32)

    def save_stage1(self, filepath: str):
        """Save Stage 1 data to NPZ"""
        if self.grad_h is None:
            return
        np.savez(filepath,
                 grad_h=self.grad_h,
                 grad_v=self.grad_v,
                 grad_magnitude=self.grad_magnitude)

    def load_stage1(self, filepath: str):
        """Load Stage 1 data from NPZ"""
        try:
            data = np.load(filepath)
            self.grad_h = data['grad_h']
            self.grad_v = data['grad_v']
            self.grad_magnitude = data['grad_magnitude']
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Stage 2: Window Selector
    # -------------------------------------------------------------------------

    def set_stage2_window(self, win_size: np.ndarray, grad_used: np.ndarray):
        """Set Stage 2 window selector outputs"""
        self.win_size = win_size.astype(np.uint8)
        self.grad_used = grad_used.astype(np.uint32)

    def save_stage2(self, filepath: str):
        """Save Stage 2 data to NPZ"""
        if self.win_size is None:
            return
        np.savez(filepath,
                 win_size=self.win_size,
                 grad_used=self.grad_used)

    def load_stage2(self, filepath: str):
        """Load Stage 2 data from NPZ"""
        try:
            data = np.load(filepath)
            self.win_size = data['win_size']
            self.grad_used = data['grad_used']
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Stage 3: Directional Filter
    # -------------------------------------------------------------------------

    def set_stage3_directional(self, avg_c: np.ndarray, avg_u: np.ndarray,
                               avg_d: np.ndarray, avg_l: np.ndarray,
                               avg_r: np.ndarray, blend0_avg: np.ndarray,
                               blend1_avg: np.ndarray):
        """Set Stage 3 directional filter outputs"""
        self.avg_c = avg_c.astype(np.uint16)
        self.avg_u = avg_u.astype(np.uint16)
        self.avg_d = avg_d.astype(np.uint16)
        self.avg_l = avg_l.astype(np.uint16)
        self.avg_r = avg_r.astype(np.uint16)
        self.blend0_avg = blend0_avg.astype(np.uint16)
        self.blend1_avg = blend1_avg.astype(np.uint16)

    def save_stage3(self, filepath: str):
        """Save Stage 3 data to NPZ"""
        if self.avg_c is None:
            return
        np.savez(filepath,
                 avg_c=self.avg_c,
                 avg_u=self.avg_u,
                 avg_d=self.avg_d,
                 avg_l=self.avg_l,
                 avg_r=self.avg_r,
                 blend0_avg=self.blend0_avg,
                 blend1_avg=self.blend1_avg)

    def load_stage3(self, filepath: str):
        """Load Stage 3 data from NPZ"""
        try:
            data = np.load(filepath)
            self.avg_c = data['avg_c']
            self.avg_u = data['avg_u']
            self.avg_d = data['avg_d']
            self.avg_l = data['avg_l']
            self.avg_r = data['avg_r']
            self.blend0_avg = data['blend0_avg']
            self.blend1_avg = data['blend1_avg']
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Stage 4: Blending
    # -------------------------------------------------------------------------

    def set_stage4_blending(self, blend0_iir: np.ndarray, blend1_iir: np.ndarray,
                           final_output: np.ndarray):
        """Set Stage 4 blending outputs"""
        self.blend0_iir = blend0_iir.astype(np.uint16)
        self.blend1_iir = blend1_iir.astype(np.uint16)
        self.final_output = final_output.astype(np.uint16)

    def save_stage4(self, filepath: str):
        """Save Stage 4 data to NPZ"""
        if self.blend0_iir is None:
            return
        np.savez(filepath,
                 blend0_iir=self.blend0_iir,
                 blend1_iir=self.blend1_iir,
                 final_output=self.final_output)

    def load_stage4(self, filepath: str):
        """Load Stage 4 data from NPZ"""
        try:
            data = np.load(filepath)
            self.blend0_iir = data['blend0_iir']
            self.blend1_iir = data['blend1_iir']
            self.final_output = data['final_output']
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Input/Output
    # -------------------------------------------------------------------------

    def set_input(self, y: np.ndarray, u: np.ndarray, v: np.ndarray):
        """Set input YUV data"""
        self.input_data = np.stack([y, u, v], axis=-1).astype(np.uint16)

    def set_output(self, y: np.ndarray, u: np.ndarray, v: np.ndarray):
        """Set output YUV data"""
        self.output_data = np.stack([y, u, v], axis=-1).astype(np.uint16)

    def save_input(self, filepath: str):
        """Save input data to NPZ"""
        if self.input_data is None:
            return False
        np.savez(filepath, input_data=self.input_data)
        return True

    def load_input(self, filepath: str):
        """Load input data from NPZ"""
        try:
            data = np.load(filepath)
            self.input_data = data['input_data']
            return True
        except FileNotFoundError:
            return False

    def save_output(self, filepath: str):
        """Save output data to NPZ"""
        if self.output_data is None:
            return
        np.savez(filepath, output_data=self.output_data)

    def load_output(self, filepath: str):
        """Load output data from NPZ"""
        try:
            data = np.load(filepath)
            self.output_data = data['output_data']
            return True
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Save/Load All
    # -------------------------------------------------------------------------

    def save_all(self, output_dir: str):
        """Save all pattern data to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_file = output_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save all stages
        self.save_input(str(output_path / "input.npz"))
        self.save_stage1(str(output_path / "stage1_sobel.npz"))
        self.save_stage2(str(output_path / "stage2_window_selector.npz"))
        self.save_stage3(str(output_path / "stage3_directional_filter.npz"))
        self.save_stage4(str(output_path / "stage4_blending.npz"))
        self.save_output(str(output_path / "output.npz"))

        print(f"Pattern data saved to: {output_dir}")

    def load_all(self, input_dir: str):
        """Load all pattern data from directory"""
        input_path = Path(input_dir)

        # Load config
        config_file = input_path / "config.json"
        with open(config_file, 'r') as f:
            self.config = PatternConfig.from_dict(json.load(f))

        # Load all stages
        self.load_input(str(input_path / "input.npz"))
        self.load_stage1(str(input_path / "stage1_sobel.npz"))
        self.load_stage2(str(input_path / "stage2_window_selector.npz"))
        self.load_stage3(str(input_path / "stage3_directional_filter.npz"))
        self.load_stage4(str(input_path / "stage4_blending.npz"))
        self.load_output(str(input_path / "output.npz"))

        print(f"Pattern data loaded from: {input_dir}")


# =============================================================================
# Pattern Comparison Utilities
# =============================================================================

class PatternComparator:
    """
    Compare pattern data between Python and C++ implementations.

    Generates detailed reports for each pipeline stage.
    """

    def __init__(self, python_dir: str, cpp_dir: str):
        self.python_data = PatternData(PatternConfig(0, 0))
        self.cpp_data = PatternData(PatternConfig(0, 0))

        self.python_dir = python_dir
        self.cpp_dir = cpp_dir

    def load(self):
        """Load both pattern data sets"""
        self.python_data.load_all(self.python_dir)
        self.cpp_data.load_all(self.cpp_dir)

    def compare_stage1(self) -> Dict:
        """Compare Stage 1 (Sobel) outputs"""
        results = {}

        for name in ['grad_h', 'grad_v', 'grad_magnitude']:
            py_val = getattr(self.python_data, name)
            cpp_val = getattr(self.cpp_data, name)

            if py_val is None or cpp_val is None:
                results[name] = {'status': 'missing', 'error': 'Data not available'}
                continue

            diff = np.abs(py_val.astype(np.int64) - cpp_val.astype(np.int64))
            results[name] = {
                'status': 'ok',
                'max_error': int(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'mismatch_count': int(np.sum(diff > 0)),
                'total_count': int(diff.size)
            }

        return results

    def compare_stage2(self) -> Dict:
        """Compare Stage 2 (Window Selector) outputs"""
        results = {}

        for name in ['win_size', 'grad_used']:
            py_val = getattr(self.python_data, name)
            cpp_val = getattr(self.cpp_data, name)

            if py_val is None or cpp_val is None:
                results[name] = {'status': 'missing', 'error': 'Data not available'}
                continue

            diff = np.abs(py_val.astype(np.int64) - cpp_val.astype(np.int64))
            results[name] = {
                'status': 'ok',
                'max_error': int(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'mismatch_count': int(np.sum(diff > 0)),
                'total_count': int(diff.size)
            }

        # Win size distribution comparison
        if self.python_data.win_size is not None and self.cpp_data.win_size is not None:
            py_dist = np.bincount(self.python_data.win_size.flatten(), minlength=6)
            cpp_dist = np.bincount(self.cpp_data.win_size.flatten(), minlength=6)
            results['win_size_distribution'] = {
                'python': py_dist.tolist(),
                'cpp': cpp_dist.tolist()
            }

        return results

    def compare_stage3(self) -> Dict:
        """Compare Stage 3 (Directional Filter) outputs"""
        results = {}

        for name in ['avg_c', 'avg_u', 'avg_d', 'avg_l', 'avg_r',
                     'blend0_avg', 'blend1_avg']:
            py_val = getattr(self.python_data, name)
            cpp_val = getattr(self.cpp_data, name)

            if py_val is None or cpp_val is None:
                results[name] = {'status': 'missing', 'error': 'Data not available'}
                continue

            diff = np.abs(py_val.astype(np.int64) - cpp_val.astype(np.int64))
            results[name] = {
                'status': 'ok',
                'max_error': int(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'mismatch_count': int(np.sum(diff > 0)),
                'total_count': int(diff.size)
            }

        return results

    def compare_stage4(self) -> Dict:
        """Compare Stage 4 (Blending) outputs"""
        results = {}

        for name in ['blend0_iir', 'blend1_iir', 'final_output']:
            py_val = getattr(self.python_data, name)
            cpp_val = getattr(self.cpp_data, name)

            if py_val is None or cpp_val is None:
                results[name] = {'status': 'missing', 'error': 'Data not available'}
                continue

            diff = np.abs(py_val.astype(np.int64) - cpp_val.astype(np.int64))
            results[name] = {
                'status': 'ok',
                'max_error': int(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'mismatch_count': int(np.sum(diff > 0)),
                'total_count': int(diff.size)
            }

        return results

    def compare_output(self) -> Dict:
        """Compare final output"""
        results = {}

        if self.python_data.output_data is None or self.cpp_data.output_data is None:
            return {'status': 'missing', 'error': 'Output data not available'}

        py_out = self.python_data.output_data
        cpp_out = self.cpp_data.output_data

        for ch, name in enumerate(['Y', 'U', 'V']):
            diff = np.abs(py_out[:,:,ch].astype(np.int64) - cpp_out[:,:,ch].astype(np.int64))
            pixel_max = (1 << self.python_data.config.pixel_bits) - 1
            mse = np.mean(diff.astype(np.float64) ** 2)
            psnr = float('inf') if mse == 0 else 10 * np.log10((pixel_max ** 2) / mse)

            results[name] = {
                'status': 'ok',
                'max_error': int(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'mse': float(mse),
                'psnr_db': float(psnr) if psnr != float('inf') else 'INF',
                'mismatch_count': int(np.sum(diff > 0)),
                'total_count': int(diff.size)
            }

        return results

    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive comparison report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CSIIR Pattern Comparison Report")
        lines.append("=" * 80)
        lines.append(f"Python data: {self.python_dir}")
        lines.append(f"C++ data:    {self.cpp_dir}")
        lines.append("")

        # Stage 1
        lines.append("-" * 80)
        lines.append("Stage 1: Sobel Filter")
        lines.append("-" * 80)
        stage1 = self.compare_stage1()
        for name, result in stage1.items():
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'ok':
                    lines.append(f"  {name}: max_err={result['max_error']}, "
                               f"mean_err={result['mean_error']:.4f}, "
                               f"mismatches={result['mismatch_count']}")
                else:
                    lines.append(f"  {name}: {result['error']}")
        lines.append("")

        # Stage 2
        lines.append("-" * 80)
        lines.append("Stage 2: Window Selector")
        lines.append("-" * 80)
        stage2 = self.compare_stage2()
        for name, result in stage2.items():
            if name == 'win_size_distribution':
                continue
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'ok':
                    lines.append(f"  {name}: max_err={result['max_error']}, "
                               f"mean_err={result['mean_error']:.4f}")
                else:
                    lines.append(f"  {name}: {result['error']}")

        if 'win_size_distribution' in stage2:
            dist = stage2['win_size_distribution']
            lines.append(f"  Window size distribution:")
            lines.append(f"    Python: {dist['python']}")
            lines.append(f"    C++:    {dist['cpp']}")
        lines.append("")

        # Stage 3
        lines.append("-" * 80)
        lines.append("Stage 3: Directional Filter")
        lines.append("-" * 80)
        stage3 = self.compare_stage3()
        for name, result in stage3.items():
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'ok':
                    lines.append(f"  {name}: max_err={result['max_error']}, "
                               f"mean_err={result['mean_error']:.4f}")
                else:
                    lines.append(f"  {name}: {result['error']}")
        lines.append("")

        # Stage 4
        lines.append("-" * 80)
        lines.append("Stage 4: Blending")
        lines.append("-" * 80)
        stage4 = self.compare_stage4()
        for name, result in stage4.items():
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'ok':
                    lines.append(f"  {name}: max_err={result['max_error']}, "
                               f"mean_err={result['mean_error']:.4f}")
                else:
                    lines.append(f"  {name}: {result['error']}")
        lines.append("")

        # Final Output
        lines.append("-" * 80)
        lines.append("Final Output Comparison")
        lines.append("-" * 80)
        output = self.compare_output()
        for name, result in output.items():
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'ok':
                    lines.append(f"  {name} Channel:")
                    lines.append(f"    PSNR:    {result['psnr_db']} dB")
                    lines.append(f"    MSE:     {result['mse']:.4f}")
                    lines.append(f"    Max Err: {result['max_error']}")
                    lines.append(f"    Mean Err: {result['mean_error']:.4f}")
                else:
                    lines.append(f"  {name}: {result['error']}")
        lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")

        return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Demo/test entry point"""
    print("=" * 60)
    print("CSIIR Pattern Data Format Test")
    print("=" * 60)

    # Create test pattern
    config = PatternConfig(
        width=64,
        height=64,
        pixel_bits=10,
        model="test"
    )

    pattern = PatternData(config)

    # Generate test data
    h, w = 64, 64
    pattern.set_input(
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w))
    )
    pattern.set_stage1_sobel(
        np.random.randint(-1000, 1000, (h, w)),
        np.random.randint(-1000, 1000, (h, w)),
        np.random.randint(0, 2000, (h, w))
    )

    pattern.set_stage2_window(
        np.random.randint(2, 6, (h, w)),
        np.random.randint(0, 2000, (h, w))
    )

    pattern.set_stage3_directional(
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w))
    )

    pattern.set_stage4_blending(
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w))
    )

    pattern.set_output(
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w)),
        np.random.randint(0, 1023, (h, w))
    )

    # Save and load
    test_dir = "/tmp/test_pattern"
    pattern.save_all(test_dir)

    loaded = PatternData(PatternConfig(0, 0))
    loaded.load_all(test_dir)

    print(f"\nLoaded config: {loaded.config}")
    print(f"grad_h shape: {loaded.grad_h.shape if loaded.grad_h is not None else 'None'}")

    print("\n" + "=" * 60)
    print("Pattern format test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()