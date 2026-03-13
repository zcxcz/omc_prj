/**
 * @file csiir_top.h
 * @brief CSIIR 顶层模块头文件
 *
 * HLS CSIIR 模块 - 顶层接口
 *
 * @version 1.0
 * @date 2026-03-13
 */

#ifndef CSIIR_TOP_H
#define CSIIR_TOP_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief CSIIR 顶层模块
 *
 * 功能:
 * - 完整的 CSIIR 4 阶段流水线处理
 * - AXI-Stream 接口
 * - 可配置参数
 *
 * 流水线:
 * Stage 1: Sobel 5x5 梯度计算
 * Stage 2: 窗口大小选择 + 方向平均滤波
 * Stage 3: 梯度加权平均
 * Stage 4: IIR Blending + 最终融合
 */
void csiir_top(
    // AXI-Stream 输入
    hls::stream<AxisUV> &axis_in,

    // AXI-Stream 输出
    hls::stream<AxisUV> &axis_out,

    // 配置参数
    CSIIRConfig &config,

    // 图像尺寸
    index_t width,
    index_t height
);

/**
 * @brief 单通道 CSIIR 处理
 */
void csiir_process_channel(
    hls::stream<pixel_t> &pixel_in,
    hls::stream<ap_uint<1>> &last_in,
    hls::stream<pixel_t> &pixel_out,
    hls::stream<ap_uint<1>> &last_out,
    CSIIRConfig &config,
    index_t width,
    index_t height
);

#endif // CSIIR_TOP_H