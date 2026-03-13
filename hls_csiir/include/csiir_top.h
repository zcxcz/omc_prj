/**
 * @file csiir_top.h
 * @brief CSIIR 顶层模块头文件
 *
 * HLS CSIIR 模块 - 顶层接口
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持 YUV 三通道独立处理
 * - 支持 8K 分辨率
 * - 支持 10-bit 像素
 */

#ifndef CSIIR_TOP_H
#define CSIIR_TOP_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief CSIIR UV 双通道处理 (YUV422 兼容模式)
 *
 * 功能:
 * - 完整的 CSIIR 4 阶段流水线处理
 * - AXI-Stream 接口
 * - 可配置参数
 * - 向后兼容原有 UV 双通道处理
 *
 * 流水线:
 * Stage 1: Sobel 5x5 梯度计算
 * Stage 2: 窗口大小选择 + 方向平均滤波
 * Stage 3: 梯度加权平均
 * Stage 4: IIR Blending + 最终融合
 */
void csiir_top_uv(
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
 * @brief CSIIR YUV 三通道独立处理
 *
 * 功能:
 * - Y/U/V 各通道独立进行 CSIIR 滤波
 * - 支持全分辨率处理
 * - 保持通道间独立性
 *
 * 数据流:
 * - 输入: AxisYUV (Y, U, V 独立传输)
 * - 内部: 三通道并行 CSIIR 处理
 * - 输出: AxisYUV (处理后 Y, U, V)
 */
void csiir_top_yuv(
    // AXI-Stream 输入
    hls::stream<AxisYUV> &axis_in,

    // AXI-Stream 输出
    hls::stream<AxisYUV> &axis_out,

    // 配置参数
    CSIIRConfig &config,

    // 图像尺寸
    index_t width,
    index_t height
);

/**
 * @brief CSIIR 顶层模块 (格式自适应)
 *
 * 根据 YUV_FORMAT 配置自动选择处理模式:
 * - YUV444: 使用 csiir_top_yuv 三通道独立处理
 * - YUV422/420: 使用 AxisYUV422 格式
 */
void csiir_top(
    // AXI-Stream 输入
    hls::stream<AxisYUV> &axis_in,

    // AXI-Stream 输出
    hls::stream<AxisYUV> &axis_out,

    // 配置参数
    CSIIRConfig &config,

    // 图像尺寸
    index_t width,
    index_t height
);

/**
 * @brief 单通道 CSIIR 处理核心
 *
 * Y/U/V 通道均可调用此函数进行独立处理
 * 实现 CSIIR 4 阶段流水线
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