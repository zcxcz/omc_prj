/**
 * @file blending.h
 * @brief Blending 模块头文件
 *
 * HLS CSIIR 模块 - Stage 4: IIR Blending 和最终融合
 *
 * @version 1.0
 * @date 2026-03-13
 */

#ifndef BLENDING_H
#define BLENDING_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief IIR Blending 模块
 *
 * 功能:
 * - 将当前方向平均与上一行数据进行 IIR 融合
 * - 融合系数根据窗口大小可配置
 *
 * 公式: output = ratio * current + (64 - ratio) * prev / 64
 */
pixel_t iir_blend(
    pixel_t current,
    pixel_t prev,
    winsize_t win_size,
    ap_uint<8> blend_ratio[4]
);

/**
 * @brief 应用 Blend Factor
 *
 * 公式: blend = (iir_avg * factor + src * (16 - factor)) / 16
 */
void apply_blend_factor(
    pixel_t iir_avg,
    ap_uint<5> factor,
    pixel_t src,
    pixel_t &blend_out
);

/**
 * @brief 最终融合
 *
 * 将 blend0 和 blend1 按照 win_size 权重融合
 *
 * blend0_weight = 2 * idx + 1 (idx = win_size / 8 - 2)
 * blend1_weight = 7 - 2 * idx
 */
pixel_t final_blend(
    pixel_t blend0,
    pixel_t blend1,
    winsize_t win_size
);

/**
 * @brief Blending 流水线模块
 *
 * 完整的 Blending 处理流程
 */
void blending_pipeline(
    hls::stream<pixel_t> &blend0_avg_in,
    hls::stream<pixel_t> &blend1_avg_in,
    hls::stream<pixel_t> &avg0_u_in,
    hls::stream<pixel_t> &avg1_u_in,
    hls::stream<winsize_t> &win_size_in,
    hls::stream<pixel_t> &pixel_in,
    hls::stream<ap_uint<1>> &last_in,
    hls::stream<pixel_t> &output_out,
    hls::stream<ap_uint<1>> &last_out,
    ap_uint<8> blend_ratio[4],
    index_t width
);

#endif // BLENDING_H