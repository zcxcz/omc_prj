/**
 * @file sobel_filter.h
 * @brief Sobel 5x5 滤波器模块头文件
 *
 * HLS CSIIR 模块 - Stage 1: Sobel 滤波
 *
 * @version 1.0
 * @date 2026-03-13
 */

#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief Sobel 5x5 滤波器模块
 *
 * 功能:
 * - 从输入流读取 5x5 窗口数据
 * - 计算 Sobel_X 和 Sobel_Y 卷积
 * - 输出梯度幅度 |Gx| + |Gy|
 *
 * Pipeline: II=1 (每时钟周期处理1像素)
 */
void sobel_filter_5x5(
    hls::stream<pixel_t> &pixel_in,     ///< 输入像素流
    hls::stream<ap_uint<1>> &last_in,   ///< 行结束标志
    hls::stream<grad_signed_t> &grad_h_out,  ///< 水平梯度输出
    hls::stream<grad_signed_t> &grad_v_out,  ///< 垂直梯度输出
    hls::stream<grad_t> &grad_out,      ///< 梯度幅度输出
    hls::stream<pixel_t> &pixel_out,    ///< 像素直通输出
    hls::stream<ap_uint<1>> &last_out,  ///< 标志直通输出
    index_t width,                       ///< 图像宽度
    index_t height                       ///< 图像高度
);

/**
 * @brief Sobel 5x5 滤波器 (窗口输入版本)
 *
 * 直接使用 5x5 窗口作为输入，用于内部模块调用
 */
void sobel_filter_5x5_window(
    pixel_t window[5][5],               ///< 5x5 输入窗口
    grad_signed_t &grad_h,              ///< 水平梯度输出
    grad_signed_t &grad_v,              ///< 垂直梯度输出
    grad_t &grad                        ///< 梯度幅度输出
);

#endif // SOBEL_FILTER_H