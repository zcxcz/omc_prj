/**
 * @file csiir_types.h
 * @brief CSIIR 模块数据类型定义
 *
 * HLS CSIIR (Color Space IIR) 模块 - 数据类型定义
 *
 * @version 1.0
 * @date 2026-03-09
 */

#ifndef CSIIR_TYPES_H
#define CSIIR_TYPES_H

// 检测是否在 Vivado HLS 环境中
#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#else
// 使用模拟类型进行独立编译测试
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#endif

//=============================================================================
// 基础数据类型定义
//=============================================================================

/**
 * @brief 像素类型 (8-bit unsigned)
 * 用于 UV 像素值，范围 [0, 255]
 */
typedef ap_uint<8>   pixel_t;

/**
 * @brief 有符号梯度类型 (16-bit signed)
 * 用于 Sobel 卷积输出 Gx, Gy
 * 范围: [-18360, +18360]
 */
typedef ap_int<16>   grad_signed_t;

/**
 * @brief 梯度幅度类型 (16-bit unsigned)
 * 用于梯度幅度 |Gx| + |Gy|
 * 范围: [0, 36720]
 */
typedef ap_uint<16>  grad_t;

/**
 * @brief 窗口大小类型 (3-bit)
 * 编码: 2=2x2, 3=3x3, 4=4x4, 5=5x5
 */
typedef ap_uint<3>   winsize_t;

/**
 * @brief 系数类型 (8-bit unsigned)
 * 用于可配置系数，范围 [0, 255]
 */
typedef ap_uint<8>   coeff_t;

/**
 * @brief 通用累加器类型 (32-bit unsigned)
 * 用于加权求和、除法运算中间值
 */
typedef ap_uint<32>  acc_t;

/**
 * @brief 索引类型 (16-bit unsigned)
 * 用于行/列索引，最大支持 65535
 */
typedef ap_uint<16>  index_t;

//=============================================================================
// 配置结构体
//=============================================================================

/**
 * @brief CSIIR 模块配置参数
 */
struct CSIIRConfig {
    // Sobel 阈值 (用于窗口大小选择)
    ap_uint<8>  sobel_thresh_0;      // 2x2 窗口阈值
    ap_uint<8>  sobel_thresh_1;      // 3x3 窗口阈值
    ap_uint<8>  sobel_thresh_2;      // 4x4 窗口阈值
    ap_uint<8>  sobel_thresh_3;      // 5x5 窗口阈值

    // Blending 系数 (按 winSize 索引: 0=2x2, 1=3x3, 2=4x4, 3=5x5)
    ap_uint<8>  blend_coeff[4];

    // 融合系数 - 2x2 窗口 (3x3 内圈, 9个系数)
    coeff_t     fusion_coeffs_2x2[9];

    // 融合系数 - 3x3 窗口 (3x3 内圈, 9个系数)
    coeff_t     fusion_coeffs_3x3[9];

    // 融合系数 - 4x4 窗口 (5x5 全窗口, 25个系数)
    coeff_t     fusion_coeffs_4x4[25];

    // 融合系数 - 5x5 窗口 (5x5 全窗口, 25个系数)
    coeff_t     fusion_coeffs_5x5[25];
};

//=============================================================================
// AXI-Stream 数据结构
//=============================================================================

/**
 * @brief AXI-Stream 像素数据结构
 * 用于 UV 双通道传输
 */
struct AxisUV {
    pixel_t     u;              // U 分量
    pixel_t     v;              // V 分量
    ap_uint<1>  user;           // 帧开始标志 (SOF)
    ap_uint<1>  last;           // 行结束标志 (EOL)
};

/**
 * @brief 内部流数据结构 - Sobel 输出
 */
struct SobelOutput {
    pixel_t     pixel_u;        // U 像素值
    pixel_t     pixel_v;        // V 像素值
    grad_t      grad_u;         // U 通道梯度
    grad_t      grad_v;         // V 通道梯度
    ap_uint<1>  last;           // 行结束标志
};

/**
 * @brief 内部流数据结构 - 窗口选择输出
 */
struct WindowSelectOutput {
    pixel_t     pixel_u;
    pixel_t     pixel_v;
    grad_t      grad_u;
    grad_t      grad_v;
    winsize_t   win_size_u;     // U 通道窗口大小
    winsize_t   win_size_v;     // V 通道窗口大小
    ap_uint<1>  last;
};

/**
 * @brief 内部流数据结构 - 平均滤波输出
 */
struct AvgFilterOutput {
    pixel_t     avg_u;          // U 通道平均输出
    pixel_t     avg_v;          // V 通道平均输出
    winsize_t   win_size_u;
    winsize_t   win_size_v;
    ap_uint<1>  last;
};

//=============================================================================
// 常量定义
//=============================================================================

// Line Buffer 深度
#define UV_LINE_BUFFER_ROWS      6
#define GRAD_LINE_BUFFER_ROWS    5

// 最大图像尺寸
#define MAX_IMAGE_WIDTH          1920
#define MAX_IMAGE_HEIGHT         1080

// 窗口大小常量
#define WINSIZE_2x2              2
#define WINSIZE_3x3              3
#define WINSIZE_4x4              4
#define WINSIZE_5x5              5

//=============================================================================
// Sobel Kernel 定义
//=============================================================================

// Sobel X Kernel (5x5)
// 归一化系数: /24 (或移位近似)
static const ap_int<5> SOBEL_X[5][5] = {
    {-1, -2,  0,  2,  1},
    {-4, -8,  0,  8,  4},
    {-6,-12,  0, 12,  6},
    {-4, -8,  0,  8,  4},
    {-1, -2,  0,  2,  1}
};

// Sobel Y Kernel (5x5)
static const ap_int<5> SOBEL_Y[5][5] = {
    {-1, -4, -6, -4, -1},
    {-2, -8,-12, -8, -2},
    { 0,  0,  0,  0,  0},
    { 2,  8, 12,  8,  2},
    { 1,  4,  6,  4,  1}
};

#endif // CSIIR_TYPES_H