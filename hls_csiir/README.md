# HLS CSIIR C-Simulation

## 编译说明

使用 Vivado HLS 或 g++ 进行 C-Simulation。

### 方法 1: Vivado HLS

```tcl
# 打开 Vivado HLS
vivado_hls -p hls_csiir

# 在 TCL 控制台执行
open_project hls_csiir
set_top csiir_top
add_files src/csiir_types.h
add_files src/sobel_filter.cpp
add_files src/window_selector.cpp
add_files src/directional_filter.cpp
add_files src/blending.cpp
add_files src/csiir_top.cpp
add_files -tb tb/tb_csiir.cpp
open_solution "solution1"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
csim_design
```

### 方法 2: g++ 编译测试

```bash
cd hls_csiir
g++ -std=c++11 -I./include -I$XILINX_HLS/include \
    src/*.cpp tb/tb_csiir.cpp -o tb_csiir
./tb_csiir
```

## 预期输出

```
============================================================
CSIIR Module Testbench
============================================================

Test configuration:
  Image size: 64 x 64
  Thresholds: 16, 24, 32, 40

Generating test input...
Input pixels: 4096

Running CSIIR...

Verifying output...
Total pixels output: 4096

============================================================
TEST PASSED
============================================================
```

## HLS Directives (待优化)

```tcl
# Pipeline 优化
set_directive_pipeline "sobel_filter_5x5/row_col"
set_directive_pipeline "window_selector/row_col"
set_directive_pipeline "csiir_process_channel/row_col"

# Array Partition
set_directive_array_partition -dim 2 -type cyclic -factor 4 "pixel_buf"
set_directive_array_partition -dim 2 -type cyclic -factor 4 "grad_buf"

# Inline
set_directive_inline "sobel_filter_5x5_window"
set_directive_inline "select_window_size"
```