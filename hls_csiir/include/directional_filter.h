/**
 * @file directional_filter.h
 * @brief 方向平均滤波模块头文件
 *
 * HLS CSIIR 模块 - Stage 2/3: 方向平均滤波
 *
 * @version 1.0
 * @date 2026-03-13
 */

#ifndef DIRECTIONAL_FILTER_H
#define DIRECTIONAL_FILTER_H

#include "csiir_types.h"
#include <hls_stream.h>

/**
 * @brief 方向平均滤波输出结构
 */
struct DirFilterOutput {
    pixel_t avg_c;      ///< 中心平均值
    pixel_t avg_u;      ///< 上方平均值
    pixel_t avg_d;      ///< 下方平均值
    pixel_t avg_l;      ///< 左侧平均值
    pixel_t avg_r;      ///< 右侧平均值
};

/**
 * @brief 方向平均滤波模块
 *
 * 功能:
 * - 计算 5 方向加权平均 (center, up, down, left, right)
 * - 根据窗口大小选择不同的权重矩阵
 * - 输出方向平均值供后续梯度加权使用
 */
void directional_filter(
    pixel_t window[5][5],           ///< 5x5 输入窗口
    winsize_t win_size,             ///< 窗口大小
    DirFilterOutput &avg0_out,      ///< avg0 输出 (较小窗口)
    DirFilterOutput &avg1_out       ///< avg1 输出 (较大窗口)
);

/**
 * @brief 梯度加权平均
 *
 * 使用方向梯度对方向平均值进行加权
 */
pixel_t gradient_weighted_avg(
    DirFilterOutput &avgs,
    grad_t grad_c,
    grad_t grad_u,
    grad_t grad_d,
    grad_t grad_l,
    grad_t grad_r
);

#endif // DIRECTIONAL_FILTER_H