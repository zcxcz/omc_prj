/**
 * @file sobel_filter.cpp
 * @brief Sobel 5x5 滤波器模块实现
 *
 * HLS CSIIR 模块 - Stage 1: Sobel 滤波
 *
 * @version 1.0
 * @date 2026-03-13
 */

#include "sobel_filter.h"

// Sobel kernels are defined in csiir_types.h

void sobel_filter_5x5_window(
    pixel_t window[5][5],
    grad_signed_t &grad_h,
    grad_signed_t &grad_v,
    grad_t &grad)
{
    // 计算 Sobel X 卷积
    ap_int<32> sum_x = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            sum_x += (ap_int<32>)window[i][j] * (ap_int<32>)SOBEL_X[i][j];
        }
    }

    // 计算 Sobel Y 卷积
    ap_int<32> sum_y = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            sum_y += (ap_int<32>)window[i][j] * (ap_int<32>)SOBEL_Y[i][j];
        }
    }

    // 截断到 16-bit
    grad_h = (grad_signed_t)sum_x;
    grad_v = (grad_signed_t)sum_y;

    // 梯度幅度 = |Gx|/5 + |Gy|/5
    uint64_t abs_h = (sum_x < 0) ? (uint64_t)(-sum_x) : (uint64_t)sum_x;
    uint64_t abs_v = (sum_y < 0) ? (uint64_t)(-sum_y) : (uint64_t)sum_y;

    // 除以 5 (使用近似)
    uint64_t grad_h_scaled = (abs_h + 2) / 5;
    uint64_t grad_v_scaled = (abs_v + 2) / 5;

    grad = (grad_t)(grad_h_scaled + grad_v_scaled);
}

void sobel_filter_5x5(
    hls::stream<pixel_t> &pixel_in,
    hls::stream<ap_uint<1>> &last_in,
    hls::stream<grad_signed_t> &grad_h_out,
    hls::stream<grad_signed_t> &grad_v_out,
    hls::stream<grad_t> &grad_out,
    hls::stream<pixel_t> &pixel_out,
    hls::stream<ap_uint<1>> &last_out,
    index_t width,
    index_t height)
{
    // Line Buffer: 5 行缓存
    pixel_t line_buf[5][MAX_IMAGE_WIDTH];

    // 滑动窗口
    pixel_t window[5][5];

#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // 初始化 Line Buffer
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < MAX_IMAGE_WIDTH; j++) {
#pragma HLS UNROLL
            line_buf[i][j] = 0;
        }
    }

    // 处理每个像素
    for (index_t row = 0; row < height + 2; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1

            // 读取输入像素
            pixel_t pixel_val = 0;
            ap_uint<1> last_val = 0;

            if (row < height) {
                pixel_val = pixel_in.read();
                last_val = last_in.read();
            }

            // 更新 Line Buffer (滚动)
            for (int i = 4; i > 0; i--) {
                for (int j = 0; j < MAX_IMAGE_WIDTH; j++) {
#pragma HLS UNROLL
                    if (j == col) {
                        line_buf[i][j] = line_buf[i-1][j];
                    }
                }
            }
            line_buf[0][col] = pixel_val;

            // 构建滑动窗口
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    int col_idx = col - 2 + j;
                    if (col_idx >= 0 && col_idx < width) {
                        window[i][j] = line_buf[i][col_idx];
                    } else {
                        // 边界处理: 镜像填充
                        int mirrored = (col_idx < 0) ? -col_idx : (2 * width - col_idx - 2);
                        mirrored = (mirrored < 0) ? 0 : (mirrored >= width ? width - 1 : mirrored);
                        window[i][j] = line_buf[i][mirrored];
                    }
                }
            }

            // 延迟输出 (需要 2 行延迟才能形成完整窗口)
            if (row >= 2) {
                grad_signed_t grad_h, grad_v;
                grad_t grad;

                sobel_filter_5x5_window(window, grad_h, grad_v, grad);

                grad_h_out.write(grad_h);
                grad_v_out.write(grad_v);
                grad_out.write(grad);

                // 直通输出 (中心像素)
                pixel_out.write(window[2][2]);
                last_out.write((row == height + 1) ? (ap_uint<1>)1 : (ap_uint<1>)0);
            }
        }
    }
}