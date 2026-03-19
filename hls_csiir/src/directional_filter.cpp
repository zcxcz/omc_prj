/**
 * @file directional_filter.cpp
 * @brief 方向平均滤波模块实现
 *
 * HLS CSIIR 模块 - Stage 2/3: 方向平均滤波
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持 10-bit 像素
 */

#include "directional_filter.h"

// 平均因子矩阵 (整数权重)
// 2x2 窗口 (内圈 3x3)
static const ap_uint<4> AVG_FACTOR_2x2[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 1, 2, 1, 0},
    {0, 2, 4, 2, 0},
    {0, 1, 2, 1, 0},
    {0, 0, 0, 0, 0}
};

// 3x3 窗口
static const ap_uint<4> AVG_FACTOR_3x3[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 1, 1, 1, 0},
    {0, 1, 1, 1, 0},
    {0, 1, 1, 1, 0},
    {0, 0, 0, 0, 0}
};

// 4x4 窗口
static const ap_uint<4> AVG_FACTOR_4x4[5][5] = {
    {1, 1, 2, 1, 1},
    {1, 2, 4, 2, 1},
    {2, 4, 8, 4, 2},
    {1, 2, 4, 2, 1},
    {1, 1, 2, 1, 1}
};

// 5x5 窗口
static const ap_uint<4> AVG_FACTOR_5x5[5][5] = {
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1}
};

// 全零矩阵 (用于 win_size == 2 时的 avg0)
static const ap_uint<4> AVG_FACTOR_ZEROS[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

// 方向掩码
static const ap_uint<1> MASK_U[5][5] = {
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

static const ap_uint<1> MASK_D[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1}
};

static const ap_uint<1> MASK_L[5][5] = {
    {1, 1, 1, 0, 0},
    {1, 1, 1, 0, 0},
    {1, 1, 1, 0, 0},
    {1, 1, 1, 0, 0},
    {1, 1, 1, 0, 0}
};

static const ap_uint<1> MASK_R[5][5] = {
    {0, 0, 1, 1, 1},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 1, 1}
};

/**
 * @brief 计算单方向加权平均
 */
static pixel_t compute_weighted_avg(
    pixel_t window[5][5],
    const ap_uint<4> factor[5][5],
    const ap_uint<1> mask[5][5] = NULL)
{
    acc_t sum_val = 0;
    acc_t sum_factor = 0;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
#pragma HLS UNROLL
            ap_uint<4> f = factor[i][j];
            if (mask != NULL) {
                f = f * mask[i][j];
            }
            sum_val += (acc_t)window[i][j] * (acc_t)f;
            sum_factor += (acc_t)f;
        }
    }

    if (sum_factor == 0) {
        return 0;
    }

    // 除法带舍入
    pixel_t result = (pixel_t)((sum_val + sum_factor / 2) / sum_factor);
    return result;
}

void directional_filter(
    pixel_t window[5][5],
    winsize_t win_size,
    DirFilterOutput &avg0_out,
    DirFilterOutput &avg1_out)
{
    // 根据 win_size 选择 avg0 和 avg1 的因子矩阵
    const ap_uint<4> (*avg0_factor)[5];
    const ap_uint<4> (*avg1_factor)[5];

    switch (win_size) {
        case WINSIZE_2x2:  // 2
            avg0_factor = AVG_FACTOR_ZEROS;  // avg0 不使用 (输出 0)
            avg1_factor = AVG_FACTOR_2x2;
            break;
        case WINSIZE_3x3:  // 3
            avg0_factor = AVG_FACTOR_2x2;
            avg1_factor = AVG_FACTOR_3x3;
            break;
        case WINSIZE_4x4:  // 4
            avg0_factor = AVG_FACTOR_3x3;
            avg1_factor = AVG_FACTOR_4x4;
            break;
        case WINSIZE_5x5:  // 5
        default:
            avg0_factor = AVG_FACTOR_4x4;
            avg1_factor = AVG_FACTOR_5x5;
            break;
    }

    // 计算 avg0 的 5 方向平均值
    avg0_out.avg_c = compute_weighted_avg(window, avg0_factor, NULL);
    avg0_out.avg_u = compute_weighted_avg(window, avg0_factor, MASK_U);
    avg0_out.avg_d = compute_weighted_avg(window, avg0_factor, MASK_D);
    avg0_out.avg_l = compute_weighted_avg(window, avg0_factor, MASK_L);
    avg0_out.avg_r = compute_weighted_avg(window, avg0_factor, MASK_R);

    // 计算 avg1 的 5 方向平均值
    avg1_out.avg_c = compute_weighted_avg(window, avg1_factor, NULL);
    avg1_out.avg_u = compute_weighted_avg(window, avg1_factor, MASK_U);
    avg1_out.avg_d = compute_weighted_avg(window, avg1_factor, MASK_D);
    avg1_out.avg_l = compute_weighted_avg(window, avg1_factor, MASK_L);
    avg1_out.avg_r = compute_weighted_avg(window, avg1_factor, MASK_R);
}

pixel_t gradient_weighted_avg(
    DirFilterOutput &avgs,
    grad_t grad_c,
    grad_t grad_u,
    grad_t grad_d,
    grad_t grad_l,
    grad_t grad_r)
{
    // 梯度排序 (逆序)
    grad_t grads[5] = {grad_u, grad_d, grad_l, grad_r, grad_c};
    pixel_t avgs_arr[5] = {avgs.avg_u, avgs.avg_d, avgs.avg_l, avgs.avg_r, avgs.avg_c};

    // 简单排序 (5 个元素，使用插入排序)
    for (int i = 1; i < 5; i++) {
        grad_t key = grads[i];
        pixel_t key_avg = avgs_arr[i];
        int j = i - 1;
        while (j >= 0 && grads[j] < key) {
            grads[j + 1] = grads[j];
            avgs_arr[j + 1] = avgs_arr[j];
            j--;
        }
        grads[j + 1] = key;
        avgs_arr[j + 1] = key_avg;
    }

    // 计算梯度加权和
    acc_t grad_sum = 0;
    acc_t weighted_sum = 0;

    for (int i = 0; i < 5; i++) {
        grad_sum += grads[i];
        weighted_sum += (acc_t)avgs_arr[i] * grads[i];
    }

    if (grad_sum == 0) {
        // 简单平均
        return (pixel_t)((avgs.avg_u + avgs.avg_d + avgs.avg_l + avgs.avg_r + avgs.avg_c) / 5);
    }

    // 加权平均
    return (pixel_t)((weighted_sum + grad_sum / 2) / grad_sum);
}