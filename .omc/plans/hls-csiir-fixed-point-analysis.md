# CSIIR 定点化分析报告

## 文档信息
- **版本**: v1.1
- **日期**: 2026-03-13
- **状态**: 验证完成

---

## 一、输入数据分析

### 1.1 原始数据格式

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入格式 | YUV422 (UYVY) | Y 通道忽略，仅处理 UV |
| 像素位宽 | 8-bit unsigned | 范围 [0, 255] |
| 数据类型 | `ap_uint<8>` | 无符号 8 位整数 |

### 1.2 输入数据范围

```
UV 像素值: 0 ~ 255 (8-bit)
```

---

## 二、Stage 1: Sobel 5x5 梯度计算

### 2.1 Sobel Kernel 系数

**Sobel_X (5x5):**
```
[-1, -2,  0,  2,  1]
[-4, -8,  0,  8,  4]
[-6,-12,  0, 12,  6]
[-4, -8,  0,  8,  4]
[-1, -2,  0,  2,  1]
```
系数范围: [-12, +12]

**Sobel_Y (5x5):**
```
[-1, -4, -6, -4, -1]
[-2, -8,-12, -8, -2]
[ 0,  0,  0,  0,  0]
[ 2,  8, 12,  8,  2]
[ 1,  4,  6,  4,  1]
```
系数范围: [-12, +12]

### 2.2 卷积输出范围分析

**Gx 计算:**
- 最大正值: Σ(max_pixel × max_positive_coeff) = 255 × (12+8+6+8+12+... ) ≈ 255 × 72 = 18,360
- 最大负值: Σ(min_pixel × max_negative_coeff) = 0 × (-72) = 0
- 实际范围: 约 [-18,360, +18,360]

**考虑对称性，实际最大值:**
```
最大 |Gx| ≈ 255 × 72 = 18,360
最大 |Gy| ≈ 255 × 72 = 18,360
```

**需要位宽:** 15-bit signed 可表示 [-16,384, +16,383]，稍显不足
**建议位宽:** 16-bit signed (`ap_int<16>`)

### 2.3 梯度幅度计算

**公式:** `Gradient = |Gx| + |Gy|`

**范围分析:**
```
|Gx| + |Gy| ≤ 18,360 + 18,360 = 36,720
```

**需要位宽:** 16-bit unsigned 可表示 [0, 65,535]
**建议位宽:** `ap_uint<16>` 或 `ap_int<17>` (保留符号位用于后续计算)

### 2.4 Stage 1 定点化参数

| 变量 | 类型 | 位宽 | 范围 |
|------|------|------|------|
| `pixel` | `ap_uint<8>` | 8-bit | [0, 255] |
| `sobel_coeff` | `ap_int<5>` | 5-bit | [-12, +12] |
| `Gx, Gy` | `ap_int<16>` | 16-bit | [-18,360, +18,360] |
| `Gradient` | `ap_uint<16>` | 16-bit | [0, 36,720] |

---

## 三、Stage 2: 窗口选择

### 3.1 阈值比较

**阈值范围:** 可配置，建议 [0, 255] (8-bit)

**比较逻辑:**
```
if (Gradient < Thresh_0) → winSize = 2
else if (Gradient < Thresh_1) → winSize = 3
else if (Gradient < Thresh_2) → winSize = 4
else → winSize = 5
```

**位宽需求:** 无需额外位宽，直接使用梯度值比较

### 3.2 窗口大小编码

| winSize | 编码 (3-bit) |
|---------|-------------|
| 2x2 | 010 (2) |
| 3x3 | 011 (3) |
| 4x4 | 100 (4) |
| 5x5 | 101 (5) |

**类型:** `ap_uint<3>`

---

## 四、Stage 3: 梯度加权平均滤波

### 4.1 5 方向平均值计算

**方向区域大小:**
- 对于 5x5 窗口，每个方向平均约 6 个像素
- 平均值范围: [0, 255]

**累加范围:**
```
sum_6pixels ≤ 255 × 6 = 1,530
```
**累加器位宽:** 11-bit (`ap_uint<11>`)

**平均值:** 8-bit (`ap_uint<8>`)，需除以 6

### 4.2 梯度加权和计算

**公式:**
```
weighted_sum = avg_c × Grad_c + avg_u × Grad_u + avg_d × Grad_d + avg_l × Grad_l + avg_r × Grad_r
```

**最大值:**
```
weighted_sum ≤ 255 × 36,720 × 5 ≈ 46,818,000
```

**需要位宽:** 27-bit
**建议位宽:** `ap_uint<32>` (32-bit 无符号)

### 4.3 梯度求和

```
Grad_sum ≤ 36,720 × 5 = 183,600
```

**需要位宽:** 18-bit
**建议位宽:** `ap_uint<32>` (与加权和一致)

### 4.4 最终输出

```
avg_output = weighted_sum / Grad_sum
```

**输出范围:** [0, 255]
**类型:** `ap_uint<8>`

### 4.5 Stage 3 定点化参数

| 变量 | 类型 | 位宽 | 说明 |
|------|------|------|------|
| `avg_x` | `ap_uint<8>` | 8-bit | 方向平均值 |
| `sum_6px` | `ap_uint<11>` | 11-bit | 6 像素累加 |
| `Grad_x` | `ap_uint<16>` | 16-bit | 梯度值 |
| `Grad_sum` | `ap_uint<32>` | 32-bit | 梯度和 |
| `weighted_sum` | `ap_uint<32>` | 32-bit | 加权和 |
| `avg_output` | `ap_uint<8>` | 8-bit | 输出 |

---

## 五、Stage 4: 行间 Blending

### 5.1 Blending 公式

```
Output = α × current + (1-α) × prev_line
```

**其中:**
- α = blend_coeff / 255
- blend_coeff: 8-bit [0, 255]

### 5.2 计算范围

**展开计算:**
```
temp = current × blend_coeff + prev_line × (255 - blend_coeff)
output = temp / 255
```

**最大值:**
```
temp ≤ 255 × 255 + 255 × 255 = 130,050
```

**需要位宽:** 18-bit
**建议位宽:** `ap_uint<32>` (便于除法)

### 5.3 Stage 4 定点化参数

| 变量 | 类型 | 位宽 | 说明 |
|------|------|------|------|
| `blend_coeff` | `ap_uint<8>` | 8-bit | Blending 系数 |
| `current` | `ap_uint<8>` | 8-bit | 当前行输出 |
| `prev_line` | `ap_uint<8>` | 8-bit | 上一行数据 |
| `temp` | `ap_uint<32>` | 32-bit | 中间结果 |
| `blend_output` | `ap_uint<8>` | 8-bit | 输出 |

---

## 六、Stage 5: 融合写回

### 6.1 融合公式

```
output = Σ(src_i × coeff_i) / Σ(coeff_i)
```

### 6.2 5x5 窗口融合

**最大值:**
```
sum_val ≤ 255 × 255 × 25 = 1,625,625
sum_coeff ≤ 255 × 25 = 6,375
```

**需要位宽:**
- `sum_val`: 21-bit
- `sum_coeff`: 13-bit

**建议位宽:** `ap_uint<32>` (统一)

### 6.3 Stage 5 定点化参数

| 变量 | 类型 | 位宽 | 说明 |
|------|------|------|------|
| `src_pixel` | `ap_uint<8>` | 8-bit | 源像素 |
| `coeff` | `ap_uint<8>` | 8-bit | 融合系数 |
| `sum_val` | `ap_uint<32>` | 32-bit | 加权和 |
| `sum_coeff` | `ap_uint<32>` | 32-bit | 系数和 |
| `final_output` | `ap_uint<8>` | 8-bit | 最终输出 |

---

## 七、定点化类型汇总

### 7.1 基础类型定义

```cpp
// 像素类型
typedef ap_uint<8>   pixel_t;        // UV 像素值

// 梯度类型
typedef ap_int<16>   grad_signed_t;  // Sobel 有符号输出 (Gx, Gy)
typedef ap_uint<16>  grad_t;         // 梯度幅度

// 窗口大小
typedef ap_uint<3>   winsize_t;      // 窗口大小编码

// 系数类型
typedef ap_uint<8>   coeff_t;        // 可配置系数

// 累加器类型
typedef ap_uint<32>  acc_t;          // 通用累加器

// 行列索引
typedef ap_uint<16>  index_t;        // 最大支持 65535 宽/高
```

### 7.2 定点数类型 (可选优化)

如果需要更精细的精度控制，可使用 `ap_fixed`:

```cpp
// 定点数类型
typedef ap_fixed<16, 8>  fixed_16_8_t;   // 16-bit, 8 integer, 8 fraction
typedef ap_fixed<24, 12> fixed_24_12_t;  // 24-bit, 12 integer, 12 fraction
```

**建议:** 初版使用整数类型，后续优化可引入定点数。

---

## 八、精度验证方法

### 8.1 验证策略

1. **Python 浮点参考**: 使用 float32 实现完整算法
2. **Python 定点仿真**: 使用 numpy 模拟定点行为
3. **误差度量**: PSNR, SSIM, MSE

### 8.2 可接受误差

| 指标 | 目标值 |
|------|--------|
| PSNR | ≥ 40 dB |
| MSE | ≤ 1.0 |
| 最大像素误差 | ≤ 2 |

### 8.3 定点化仿真脚本

```python
import numpy as np

def simulate_fixed_point(input_image):
    """模拟 HLS 定点化行为"""
    # Stage 1: Sobel (使用 16-bit 中间值)
    gx = np.int16(convolve2d(input_image, sobel_x, mode='same'))
    gy = np.int16(convolve2d(input_image, sobel_y, mode='same'))
    gradient = np.uint16(np.abs(gx) + np.abs(gy))  # 16-bit

    # Stage 2: 窗口选择
    win_size = np.uint8(select_window(gradient, thresholds))

    # Stage 3: 加权平均 (使用 32-bit 累加器)
    weighted_sum = np.uint32(...)
    grad_sum = np.uint32(...)
    avg_output = np.uint8(weighted_sum / grad_sum)

    # Stage 4: Blending (使用 32-bit 中间值)
    temp = np.uint32(current * alpha + prev * (255 - alpha))
    blend_output = np.uint8(temp // 255)

    # Stage 5: 融合 (使用 32-bit 累加器)
    sum_val = np.uint32(...)
    sum_coeff = np.uint32(...)
    final_output = np.uint8(sum_val // sum_coeff)

    return final_output
```

---

## 九、资源影响分析

### 9.1 位宽 vs 资源

| 位宽 | DSP 使用 | LUT 使用 | BRAM 使用 |
|------|----------|----------|-----------|
| 8-bit | 1× | 低 | 低 |
| 16-bit | 1× | 中 | 中 |
| 32-bit | 2× | 高 | 高 |

### 9.2 优化建议

1. **累加器复用**: 32-bit 累加器在多个 Stage 间复用
2. **移位代替除法**: 用右移近似除法 (temp >> 8 ≈ temp / 256)
3. **流水线寄存器**: 中间结果使用流水线寄存器降低时序压力

---

## 十、结论

### 10.1 推荐类型配置

| 阶段 | 数据类型 | 位宽 | 精度 |
|------|----------|------|------|
| 输入/输出 | `ap_uint<8>` | 8-bit | 整数 |
| Sobel 中间值 | `ap_int<16>` | 16-bit | 整数 |
| 梯度 | `ap_uint<16>` | 16-bit | 整数 |
| 累加器 | `ap_uint<32>` | 32-bit | 整数 |
| 系数 | `ap_uint<8>` | 8-bit | 整数 |

### 10.2 预期精度

- PSNR ≥ 45 dB (相对于浮点参考)
- 最大像素误差 ≤ 2
- 资源开销: 中等

---

*文档版本: v1.0*
*状态: 分析完成 - 待验证*