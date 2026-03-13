/**
 * @file window_selector.h
 * @brief 窗口大小选择模块头文件
 *
 * HLS CSIIR 模块 - Stage 2: 窗口大小选择
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持 10-bit 像素 (16-bit 阈值)
 */

#ifndef WINDOW_SELECTOR_H
#define WINDOW_SELECTOR_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief 窗口大小选择模块
 *
 * 功能:
 * - 根据梯度值选择窗口大小
 * - 使用 3 个连续像素的最大梯度进行稳定选择
 * - 输出窗口大小编码 (2, 3, 4, 5)
 *
 * Pipeline: II=1
 */
void window_selector(
    hls::stream<grad_t> &grad_in,       ///< 梯度输入流
    hls::stream<ap_uint<1>> &last_in,   ///< 行结束标志
    hls::stream<winsize_t> &win_size_out, ///< 窗口大小输出
    hls::stream<grad_t> &grad_out,      ///< 梯度直通输出
    hls::stream<ap_uint<1>> &last_out,  ///< 标志直通输出
    ap_uint<16> thresh_0,               ///< 2x2 窗口阈值 (16-bit 支持 10-bit 像素)
    ap_uint<16> thresh_1,               ///< 3x3 窗口阈值
    ap_uint<16> thresh_2,               ///< 4x4 窗口阈值
    ap_uint<16> thresh_3,               ///< 5x5 窗口阈值
    index_t width,                      ///< 图像宽度
    index_t height                      ///< 图像高度
);

/**
 * @brief 单像素窗口大小选择
 *
 * 使用当前、前一、后一像素的最大梯度
 */
winsize_t select_window_size(
    grad_t grad_curr,
    grad_t grad_prev,
    grad_t grad_next,
    ap_uint<16> thresh_0,
    ap_uint<16> thresh_1,
    ap_uint<16> thresh_2,
    ap_uint<16> thresh_3
);

#endif // WINDOW_SELECTOR_H