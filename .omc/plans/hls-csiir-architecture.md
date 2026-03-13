# HLS CSIIR 模块架构设计

## 文档信息
- **版本**: v1.0
- **创建日期**: 2026-03-09
- **状态**: 设计中
- **关联文档**: [hls-csiir-project-plan.md](./hls-csiir-project-plan.md)

---

## 一、模块层次结构

### 1.1 顶层模块划分

```
csiir_top
├── csiir_line_buffer      // 行缓存管理
│   ├── uv_line_buffer     // UV 像素行缓存 (6行)
│   └── grad_line_buffer   // 梯度行缓存 (5行)
├── csiir_sobel_5x5        // Stage 1: Sobel 5x5 梯度计算
├── csiir_window_select    // Stage 2: 窗口大小选择
├── csiir_avg_filter       // Stage 3: 梯度加权平均滤波
├── csiir_blending         // Stage 4: 行间 Blending
└── csiir_fusion_writeback // Stage 5: 融合写回
```

### 1.2 模块职责

| 模块 | 功能 | 输入 | 输出 | 延迟(行) |
|------|------|------|------|----------|
| `csiir_line_buffer` | 行数据缓存与管理 | 像素流 | 5x5 窗口数据 | 2 行 |
| `csiir_sobel_5x5` | 计算梯度幅度 | 5x5 窗口 | Gradient, Gx, Gy | 0 |
| `csiir_window_select` | 选择滤波窗口大小 | Gradient | winSize | 0 |
| `csiir_avg_filter` | 5 方向加权平均 | 窗口数据, 梯度 | avg_output | 2 行 |
| `csiir_blending` | 行间混合 | avg_output, prev_line | blend_output | 0 |
| `csiir_fusion_writeback` | 融合并写回 | blend_output, src_lbuf | final_output | 0 |

---

## 二、数据流架构

### 2.1 整体数据流

```
                              ┌──────────────────────────────────────────────────────────┐
                              │                    Line Buffer 层                        │
                              │  ┌─────────────────────┐  ┌─────────────────────┐       │
                              │  │   UV Line Buffer    │  │  Grad Line Buffer   │       │
                              │  │      (6 行)         │  │      (5 行)         │       │
                              │  └─────────┬───────────┘  └──────────┬──────────┘       │
                              └────────────│─────────────────────────│──────────────────┘
                                           │                         │
              Input                        ▼                         ▼
           YUV422 ──────────────────────────────────────────────────────────────────────►
           (UV only)          ┌─────────────────────┐
                              │   Stage 1:          │
                              │   Sobel 5x5         │───────────────────────► Grad
                              │   梯度计算          │         │
                              └─────────────────────┘         │
                                                            ▼
                                              ┌─────────────────────┐
                                              │   Stage 2:          │
                                              │   窗口大小选择       │
                                              │   (winSize 2~5)     │
                                              └─────────┬───────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                              ▼                         ▼                         ▼
                      ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
                      │  2x2 Filter   │         │  3x3 Filter   │         │  4x4/5x5      │
                      │  (可选)       │         │  (可选)       │         │  Filter       │
                      └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
                              │                         │                         │
                              └─────────────────────────┼─────────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │   Stage 3:          │
                                              │   梯度加权平均       │◄──── Grad (from LB)
                                              │   (5 方向)          │
                                              └─────────┬───────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │   Stage 4:          │
                                              │   行间 Blending     │◄──── PrevLine (from UV LB)
                                              │                     │
                                              └─────────┬───────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │   Stage 5:          │
                                              │   融合写回          │────► 写回 UV Line Buffer
                                              │                     │
                                              └─────────┬───────────┘
                                                        │
                                                        ▼
                                                   Output (UV)
```

### 2.2 Pipeline 时序

```
时间 (时钟周期) ──────────────────────────────────────────────────────────────────►

像素流:
P0    P1    P2    P3    P4    P5    P6    P7    P8    P9    ...
│     │     │     │     │     │     │     │     │     │
▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ LB  │ LB  │ LB  │ LB  │ LB  │ LB  │ LB  │ LB  │ LB  │ LB  │ ← 行缓存填充 (前2行)
│ 填充│ 填充│ 填充│ 填充│ 填充│ 填充│ 填充│ 填充│ 填充│ 填充│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    ┌─────┬─────┬─────┬─────┬─────┬─────┐
                    │Sobel│Sobel│Sobel│Sobel│Sobel│Sobel│ ← Stage 1 (II=1)
                    │ P0  │ P1  │ P2  │ P3  │ P4  │ P5  │
                    └─────┴─────┴─────┴─────┴─────┴─────┘
                    ┌─────┬─────┬─────┬─────┬─────┬─────┐
                    │ Win │ Win │ Win │ Win │ Win │ Win │ ← Stage 2 (II=1)
                    │Sel  │Sel  │Sel  │Sel  │Sel  │Sel  │
                    └─────┴─────┴─────┴─────┴─────┴─────┘
                              ┌─────┬─────┬─────┬─────┐
                              │ Avg │ Avg │ Avg │ Avg │ ← Stage 3 (延迟若干行)
                              │ P0  │ P1  │ P2  │ P3  │
                              └─────┴─────┴─────┴─────┘
                              ┌─────┬─────┬─────┬─────┐
                              │Blend│Blend│Blend│Blend│ ← Stage 4
                              │ P0  │ P1  │ P2  │ P3  │
                              └─────┴─────┴─────┴─────┘
                              ┌─────┬─────┬─────┬─────┐
                              │Write│Write│Write│Write│ ← Stage 5: 写回
                              │Back │Back │Back │Back │
                              └─────┴─────┴─────┴─────┘

总延迟: ~5-6 行 (Line Buffer 填充 + 各 Stage 处理)
吞吐量: 1 pixel/clock (Pipeline II=1)
```

---

## 三、接口定义

### 3.1 顶层接口

```cpp
// csiir_top.h
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// 数据类型定义
typedef ap_uint<8>  pixel_t;           // 8-bit 像素值
typedef ap_int<16>  grad_t;            // 16-bit 梯度值
typedef ap_uint<3>  winsize_t;         // 窗口大小编码 (2,3,4,5)
typedef ap_uint<8>  coeff_t;           // 系数 (0-255)

// 配置寄存器结构
struct CSIIRConfig {
    // Sobel 阈值
    ap_uint<8>  sobel_thresh_0;        // 2x2 窗口阈值
    ap_uint<8>  sobel_thresh_1;        // 3x3 窗口阈值
    ap_uint<8>  sobel_thresh_2;        // 4x4 窗口阈值
    ap_uint<8>  sobel_thresh_3;        // 5x5 窗口阈值

    // Blending 系数
    ap_uint<8>  blend_coeff[4];        // 按 winSize 索引

    // 融合系数 (每窗口 25 个)
    coeff_t     fusion_coeffs_2x2[9];
    coeff_t     fusion_coeffs_3x3[9];
    coeff_t     fusion_coeffs_4x4[25];
    coeff_t     fusion_coeffs_5x5[25];
};

// AXI-Stream 数据结构
struct AxisPixel {
    pixel_t     u;                     // U 分量
    pixel_t     v;                     // V 分量
    ap_uint<1>  last;                  // 行结束标志
};

// 顶层函数接口
void csiir_top(
    // 视频流接口
    hls::stream<AxisPixel>&    in_stream,    // 输入 UV 流
    hls::stream<AxisPixel>&    out_stream,   // 输出 UV 流

    // 配置接口 (AXI4-Lite)
    const CSIIRConfig&         config,

    // 帧控制
    ap_uint<16>                frame_width,   // 图像宽度
    ap_uint<16>                frame_height,  // 图像高度

    // 行同步
    ap_uint<1>&                sof,           // 帧开始
    ap_uint<1>&                eof            // 帧结束
);
```

### 3.2 内部流接口

```cpp
// Stage 间数据传递
struct SobelOutput {
    pixel_t     pixel_val;       // 原始像素值
    grad_t      gradient;        // 梯度幅度
    ap_uint<1>  last;            // 行结束标志
};

struct WindowSelectOutput {
    pixel_t     pixel_val;
    grad_t      gradient;
    winsize_t   win_size;
    ap_uint<1>  last;
};

struct AvgFilterOutput {
    pixel_t     avg_value;       // 平均滤波输出
    winsize_t   win_size;
    ap_uint<1>  last;
};
```

---

## 四、Line Buffer 详细设计

### 4.1 UV Line Buffer (6行)

```cpp
// uv_line_buffer.h

#define UV_LINE_BUFFER_DEPTH  6
#define MAX_IMAGE_WIDTH       1920

class UVLineBuffer {
public:
    // 内部存储: 使用 BRAM 实现
    pixel_t buffer[UV_LINE_BUFFER_DEPTH][MAX_IMAGE_WIDTH];

    // 写入新像素 (Row 5)
    void write_new_pixel(pixel_t val, ap_uint<16> col);

    // 读取 5x5 窗口 (Row 0-4)
    void read_window_5x5(
        ap_uint<16> col,
        pixel_t window[5][5]
    );

    // 融合写回 (Row 0-4)
    void writeback_fusion(
        pixel_t val,
        ap_uint<16> row,
        ap_uint<16> col
    );

    // 读取上一行 (用于 Blending)
    pixel_t read_prev_line(ap_uint<16> row, ap_uint<16> col);

    // 行滚动
    void shift_lines();

    // 冲突检测与反压
    bool check_conflict(ap_uint<16> col);
};
```

**HLS Directive:**
```cpp
#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=2 dim=2
#pragma HLS RESOURCE variable=buffer core=RAM_2P_BRAM
```

### 4.2 Gradient Line Buffer (5行)

```cpp
// grad_line_buffer.h

#define GRAD_LINE_BUFFER_DEPTH  5

class GradLineBuffer {
public:
    grad_t buffer[GRAD_LINE_BUFFER_DEPTH][MAX_IMAGE_WIDTH];

    // 写入梯度值
    void write_grad(grad_t val, ap_uint<16> col);

    // 读取 5 方向梯度 (上、下、左、右、中心)
    void read_5dir_gradients(
        ap_uint<16> col,
        grad_t& grad_u,    // 上
        grad_t& grad_d,    // 下
        grad_t& grad_l,    // 左
        grad_t& grad_r,    // 右
        grad_t& grad_c     // 中心
    );

    // 行滚动
    void shift_lines();
};
```

### 4.3 反压机制实现

```cpp
// backpressure_control.h

class BackpressureControl {
private:
    bool writeback_active;
    bool new_pixel_active;
    ap_uint<16> writeback_col;
    ap_uint<16> new_pixel_col;

public:
    // 检测冲突
    bool check_conflict() {
        return (writeback_active && new_pixel_active &&
                writeback_col == new_pixel_col);
    }

    // 生成反压信号
    void generate_backpressure(
        bool& stall_input,
        bool& stall_pipeline
    );
};
```

**时序示例:**
```
正常情况:
Cycle:     |  1  |  2  |  3  |  4  |  5  |
新像素写入: | W5  | W5  | W5  | W5  | W5  |  ← Row 5 正常写入
融合写回:   |     | WB0 | WB1 | WB2 | WB3 |  ← 写回 Row 0-4
冲突:       |     |     |     |     |     |  ← 无冲突

冲突情况 (同列写入):
Cycle:     |  1  |  2  |  3  |  4  |  5  |
新像素写入: | W5  | W5  | --  | W5  | W5  |  ← 第3周期暂停
融合写回:   |     | WB0 | WB0 | WB1 | WB2 |  ← 延迟一周期
反压:       |     |     |STALL|     |      |  ← 暂停前级
```

---

## 五、各 Stage HLS 实现

### 5.1 Stage 1: Sobel 5x5

```cpp
// csiir_sobel_5x5.cpp

void csiir_sobel_5x5(
    pixel_t  window[5][5],
    grad_t&  gradient
) {
    // Sobel X Kernel (5x5)
    const ap_int<8> sobel_x[5][5] = {
        {-1, -2,  0,  2,  1},
        {-4, -8,  0,  8,  4},
        {-6,-12,  0, 12,  6},
        {-4, -8,  0,  8,  4},
        {-1, -2,  0,  2,  1}
    };

    // Sobel Y Kernel (5x5)
    const ap_int<8> sobel_y[5][5] = {
        {-1, -4, -6, -4, -1},
        {-2, -8,-12, -8, -2},
        { 0,  0,  0,  0,  0},
        { 2,  8, 12,  8,  2},
        { 1,  4,  6,  4,  1}
    };

#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    ap_int<20> gx = 0;
    ap_int<20> gy = 0;

    // 卷积计算
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
#pragma HLS UNROLL
            gx += window[i][j] * sobel_x[i][j];
            gy += window[i][j] * sobel_y[i][j];
        }
    }

    // L1 范数 (硬件友好)
    gradient = (gx >= 0 ? gx : -gx) + (gy >= 0 ? gy : -gy);
}
```

**资源预估:**
- LUT: ~200 (乘法器 + 加法器)
- DSP: 0 (使用移位实现)
- 延迟: 1 cycle

### 5.2 Stage 2: 窗口选择

```cpp
// csiir_window_select.cpp

void csiir_window_select(
    grad_t       gradient,
    ap_uint<8>   thresh_0,
    ap_uint<8>   thresh_1,
    ap_uint<8>   thresh_2,
    ap_uint<8>   thresh_3,
    winsize_t&   win_size
) {
#pragma HLS PIPELINE II=1

    // 梯度比较
    if (gradient < thresh_0) {
        win_size = 2;
    } else if (gradient < thresh_1) {
        win_size = 3;
    } else if (gradient < thresh_2) {
        win_size = 4;
    } else {
        win_size = 5;
    }
}
```

**资源预估:**
- LUT: ~20 (比较器)
- 延迟: 1 cycle

### 5.3 Stage 3: 梯度加权平均滤波

```cpp
// csiir_avg_filter.cpp

void csiir_avg_filter(
    // 输入
    pixel_t      uv_window[5][5],    // UV 像素窗口
    grad_t       grad_5dir[5],       // 5 方向梯度 (u,d,l,r,c)
    winsize_t    win_size,

    // 输出
    pixel_t&     avg_output
) {
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=uv_window complete dim=0
#pragma HLS ARRAY_PARTITION variable=grad_5dir complete

    // 计算 5 方向平均值
    ap_uint<20> avg_c, avg_u, avg_d, avg_l, avg_r;

    // 中心区域
    avg_c = uv_window[2][2];

    // 上方区域 (Row 0-1)
    avg_u = (uv_window[0][1] + uv_window[0][2] + uv_window[0][3] +
             uv_window[1][1] + uv_window[1][2] + uv_window[1][3]) / 6;

    // 下方区域 (Row 3-4)
    avg_d = (uv_window[3][1] + uv_window[3][2] + uv_window[3][3] +
             uv_window[4][1] + uv_window[4][2] + uv_window[4][3]) / 6;

    // 左侧区域 (Col 0-1)
    avg_l = (uv_window[1][0] + uv_window[2][0] + uv_window[3][0] +
             uv_window[1][1] + uv_window[2][1] + uv_window[3][1]) / 6;

    // 右侧区域 (Col 3-4)
    avg_r = (uv_window[1][3] + uv_window[2][3] + uv_window[3][3] +
             uv_window[1][4] + uv_window[2][4] + uv_window[3][4]) / 6;

    // 梯度排序 (逆序)
    grad_t sorted_grad[5];
    sort_gradients(grad_5dir, sorted_grad);

    // 计算梯度加权和
    ap_uint<32> grad_sum = 0;
    ap_uint<32> weighted_sum = 0;

    for (int i = 0; i < 5; i++) {
#pragma HLS UNROLL
        grad_sum += sorted_grad[i];
    }

    if (grad_sum > 0) {
        weighted_sum = avg_c * sorted_grad[4] + avg_u * sorted_grad[3] +
                       avg_d * sorted_grad[2] + avg_l * sorted_grad[1] +
                       avg_r * sorted_grad[0];
        avg_output = weighted_sum / grad_sum;
    } else {
        avg_output = (avg_c + avg_u + avg_d + avg_l + avg_r) / 5;
    }
}

// 梯度排序 (逆序, 冒泡排序 - 5元素固定)
void sort_gradients(
    grad_t input[5],
    grad_t sorted[5]
) {
#pragma HLS INLINE
    // 5 元素冒泡排序, 完全展开
    // ...
}
```

**资源预估:**
- LUT: ~500 (排序 + 加权计算)
- DSP: ~5 (乘法器)
- 延迟: ~5 cycles

### 5.4 Stage 4: 行间 Blending

```cpp
// csiir_blending.cpp

void csiir_blending(
    pixel_t      avg_output,
    pixel_t      prev_line,
    ap_uint<8>   blend_coeff,
    winsize_t    win_size,

    pixel_t&     blend_output
) {
#pragma HLS PIPELINE II=1

    // alpha = blend_coeff / 255
    // output = alpha * avg + (1-alpha) * prev

    ap_uint<18> temp = avg_output * blend_coeff +
                       prev_line * (255 - blend_coeff);
    blend_output = temp >> 8;  // 近似除以 255
}
```

**资源预估:**
- LUT: ~50
- DSP: 2 (乘法器)
- 延迟: 1 cycle

### 5.5 Stage 5: 融合写回

```cpp
// csiir_fusion_writeback.cpp

void csiir_fusion_writeback(
    pixel_t      blend_output,
    pixel_t      src_window[5][5],
    winsize_t    win_size,
    coeff_t      fusion_coeffs[25],

    pixel_t&     final_output,
    pixel_t      writeback_value
) {
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=src_window complete dim=0
#pragma HLS ARRAY_PARTITION variable=fusion_coeffs complete

    ap_uint<32> sum_val = 0;
    ap_uint<32> sum_coeff = 0;

    // 根据窗口大小选择系数范围
    int offset = (win_size <= 3) ? 1 : 0;  // 3x3 内圈 或 5x5 全窗口
    int size = (win_size <= 3) ? 3 : 5;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
#pragma HLS UNROLL
            int idx = (win_size <= 3) ?
                      (i + offset) * 5 + (j + offset) :
                      i * 5 + j;

            sum_val += src_window[i][j] * fusion_coeffs[idx];
            sum_coeff += fusion_coeffs[idx];
        }
    }

    if (sum_coeff > 0) {
        final_output = sum_val / sum_coeff;
    } else {
        final_output = blend_output;
    }

    writeback_value = final_output;
}
```

**资源预估:**
- LUT: ~300
- DSP: ~5
- 延迟: 1 cycle

---

## 六、HLS Directive 规划

### 6.1 顶层 Directive

```cpp
// csiir_top.cpp

void csiir_top(...) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=config
#pragma HLS INTERFACE s_axilite port=frame_width
#pragma HLS INTERFACE s_axilite port=frame_height
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    // Stage 间流
    hls::stream<SobelOutput>        sobel_out;
    hls::stream<WindowSelectOutput> win_sel_out;
    hls::stream<AvgFilterOutput>    avg_out;

#pragma HLS STREAM variable=sobel_out depth=128
#pragma HLS STREAM variable=win_sel_out depth=128
#pragma HLS STREAM variable=avg_out depth=128

    // 子模块调用
    csiir_line_buffer(in_stream, ...);
    csiir_sobel_5x5(...);
    csiir_window_select(...);
    csiir_avg_filter(...);
    csiir_blending(...);
    csiir_fusion_writeback(...);
}
```

### 6.2 Line Buffer Directive

```cpp
// UV Line Buffer
#pragma HLS ARRAY_PARTITION variable=uv_buffer cyclic factor=4 dim=2
#pragma HLS RESOURCE variable=uv_buffer core=RAM_2P_URAM
#pragma HLS LATENCY min=1 max=2

// Gradient Line Buffer
#pragma HLS ARRAY_PARTITION variable=grad_buffer cyclic factor=2 dim=2
#pragma HLS RESOURCE variable=grad_buffer core=RAM_2P_BRAM
```

### 6.3 Pipeline 目标

| 模块 | 目标 II | 实际 II | 说明 |
|------|---------|---------|------|
| Line Buffer 读取 | 1 | 1 | 双端口 BRAM |
| Sobel 5x5 | 1 | 1 | 完全展开 |
| Window Select | 1 | 1 | 简单比较 |
| Avg Filter | 1 | 1-2 | 排序可能影响 |
| Blending | 1 | 1 | 简单计算 |
| Fusion Writeback | 1 | 1 | 完全展开 |

---

## 七、资源预估

### 7.1 1080p@60fps 预估

| 资源类型 | 预估使用 | Zynq UltraScale+ (ZU7EV) | 占比 |
|----------|----------|--------------------------|------|
| **LUT** | ~2,500 | 182,400 | 1.4% |
| **FF** | ~3,000 | 364,800 | 0.8% |
| **DSP** | ~15 | 1,240 | 1.2% |
| **BRAM** | ~20 | 325 | 6.2% |
| **URAM** | ~4 | 96 | 4.2% |

### 7.2 Line Buffer 资源 (1920 宽度)

| Buffer 类型 | 行数 | 位宽 | BRAM/URAM |
|-------------|------|------|-----------|
| UV Line Buffer | 6 | 8-bit × 2 (U+V) | ~2 URAM 或 12 BRAM |
| Grad Line Buffer | 5 | 16-bit | ~4 BRAM |

**总计**: ~4 URAM + 4 BRAM (使用 URAM 优化方案)

---

## 八、验证策略

### 8.1 C-Simulation 测试向量

```
test_vectors/
├── input_flat.yuv      # 平坦区域测试
├── input_edge.yuv      # 边缘区域测试
├── input_texture.yuv   # 纹理区域测试
└── reference_output/   # 参考输出 (Python 生成)
```

### 8.2 测试用例

| 用例 | 目的 | 输入特征 |
|------|------|----------|
| `test_flat` | 验证平坦区域滤波 | 全常数输入 |
| `test_edge` | 验证边缘检测 | 水平/垂直边缘 |
| `test_gradient` | 验证窗口切换 | 渐变图像 |
| `test_checkerboard` | 验证纹理处理 | 棋盘格图案 |
| `test_full_frame` | 端到端验证 | 1080p 完整帧 |

---

## 九、待确认事项

1. **定点化位宽**
   - 梯度值位宽 (建议 16-bit)
   - 中间累加器位宽 (建议 32-bit)
   - 需要定点化精度验证

2. **图像最大分辨率**
   - 影响 Line Buffer 深度
   - 影响 BRAM 资源分配

3. **目标时钟频率**
   - 200MHz? 300MHz?
   - 影响时序约束

---

## 十、文件结构

```
hls_csiir/
├── include/
│   ├── csiir_types.h          // 数据类型定义
│   ├── csiir_config.h         // 配置结构
│   └── csiir_line_buffer.h    // Line Buffer 类
├── src/
│   ├── csiir_top.cpp          // 顶层模块
│   ├── csiir_sobel_5x5.cpp    // Stage 1
│   ├── csiir_window_select.cpp // Stage 2
│   ├── csiir_avg_filter.cpp   // Stage 3
│   ├── csiir_blending.cpp     // Stage 4
│   └── csiir_fusion_writeback.cpp // Stage 5
├── tb/
│   ├── tb_csiir_top.cpp       // 顶层测试平台
│   └── test_vectors/          // 测试向量
└── directives/
    └── csiir_directives.tcl   // HLS 指令文件
```

---

*文档版本: v1.0*
*创建日期: 2026-03-09*
*状态: 设计中 - 待定点化分析*