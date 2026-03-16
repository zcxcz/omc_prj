/**
 * @file csiir_types.h
 * @brief CSIIR 模块数据类型定义
 *
 * HLS CSIIR (Color Space IIR) 模块 - 数据类型定义
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持参数化像素位宽 (8/10/12-bit)
 * - 支持 YUV 三通道独立处理
 * - 支持 8K 分辨率
 */

#ifndef CSIIR_TYPES_H
#define CSIIR_TYPES_H

// 首先包含配置文件
#include "csiir_config.h"

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
 * @brief 像素类型 (参数化位宽)
 * 默认 10-bit unsigned, 范围 [0, 1023]
 * 可通过 -DPIXEL_BITWIDTH=8 编译为 8-bit
 */
typedef ap_uint<PIXEL_BITWIDTH>  pixel_t;

/**
 * @brief 有符号梯度类型
 * 用于 Sobel 卷积输出 Gx, Gy
 *
 * 位宽计算:
 * - 10-bit 像素最大值 1023
 * - Sobel 5x5 最大系数 12 (SOBEL_Y 中心行)
 * - 最大累加: 1023 * 72 = 73,656 (使用 17-bit signed)
 * - 使用 18-bit signed 以留有余量
 */
typedef ap_int<18>   grad_signed_t;

/**
 * @brief 梯度幅度类型
 * 用于梯度幅度 |Gx| + |Gy|
 *
 * 位宽计算:
 * - 最大 |Gx| = 73,656, 最大 |Gy| = 73,656
 * - 最大 |Gx| + |Gy| = 147,312 (使用 18-bit unsigned)
 * - 使用 19-bit unsigned 以留有余量
 */
typedef ap_uint<19>  grad_t;

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
 * @brief 索引类型 (14-bit unsigned)
 * 用于行/列索引，最大支持 16383 (支持 8K = 7680x4320)
 */
typedef ap_uint<INDEX_BITWIDTH>  index_t;

//=============================================================================
// 配置结构体
//=============================================================================

/**
 * @brief CSIIR 模块配置参数
 */
struct CSIIRConfig {
    // Sobel 阈值 (用于窗口大小选择)
    // 10-bit 像素范围更大 (0-1023)，使用 16-bit 阈值
    ap_uint<16>  sobel_thresh_0;      // 2x2 窗口阈值
    ap_uint<16>  sobel_thresh_1;      // 3x3 窗口阈值
    ap_uint<16>  sobel_thresh_2;      // 4x4 窗口阈值
    ap_uint<16>  sobel_thresh_3;      // 5x5 窗口阈值

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
 * @brief AXI-Stream UV 双通道数据结构 (YUV422 兼容模式)
 * 用于 UV 双通道传输，保持向后兼容
 */
struct AxisUV {
    pixel_t     u;              // U 分量
    pixel_t     v;              // V 分量
    ap_uint<1>  user;           // 帧开始标志 (SOF)
    ap_uint<1>  last;           // 行结束标志 (EOL)
};

/**
 * @brief AXI-Stream YUV444 三通道数据结构
 * Y/U/V 三通道独立输入输出
 */
struct AxisYUV444 {
    pixel_t     y;              // Y 分量
    pixel_t     u;              // U 分量
    pixel_t     v;              // V 分量
    ap_uint<1>  user;           // 帧开始标志 (SOF)
    ap_uint<1>  last;           // 行结束标志 (EOL)
};

/**
 * @brief AXI-Stream YUV422 数据结构
 * Y 独立传输，UV 打包传输
 * 注：YUV422 时，UV 在水平方向上采样后输入，内部处理时需还原
 */
struct AxisYUV422 {
    pixel_t     y;              // Y 分量
    pixel_t     u;              // U 分量
    pixel_t     v;              // V 分量
    ap_uint<1>  user;           // 帧开始标志 (SOF)
    ap_uint<1>  last;           // 行结束标志 (EOL)
};

/**
 * @brief 根据 YUV_FORMAT 自动选择 AXI-Stream 数据类型
 */
#if YUV_FORMAT == 444
    typedef AxisYUV444 AxisYUV;
#elif YUV_FORMAT == 422 || YUV_FORMAT == 420
    typedef AxisYUV422 AxisYUV;
#endif

/**
 * @brief 内部 YUV 三通道数据
 * 用于三通道独立 CSIIR 处理
 */
struct YUVChannels {
    pixel_t     y;              // Y 分量
    pixel_t     u;              // U 分量
    pixel_t     v;              // V 分量
};

//=============================================================================
// 内部流数据结构
//=============================================================================

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
// Sobel Kernel 定义 (简化差分核 - 匹配 isp-csiir-algorithm-reference.md)
//=============================================================================

/**
 * @brief Sobel X Kernel (5x5) - 简化差分核
 * 计算上下两行的差分: 上行 - 下行
 * 归一化: /5
 */
static const ap_int<5> SOBEL_X[5][5] = {
    { 1,  1,  1,  1,  1},
    { 0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0},
    {-1, -1, -1, -1, -1}
};

/**
 * @brief Sobel Y Kernel (5x5) - 简化差分核
 * 计算左右两列的差分: 左列 - 右列
 * 归一化: /5
 */
static const ap_int<5> SOBEL_Y[5][5] = {
    { 1,  0,  0,  0, -1},
    { 1,  0,  0,  0, -1},
    { 1,  0,  0,  0, -1},
    { 1,  0,  0,  0, -1},
    { 1,  0,  0,  0, -1}
};

#endif // CSIIR_TYPES_H