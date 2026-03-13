/**
 * @file blending.cpp
 * @brief Blending 模块实现
 *
 * HLS CSIIR 模块 - Stage 4: IIR Blending 和最终融合
 *
 * @version 1.0
 * @date 2026-03-13
 */

#include "blending.h"

// Blend Factor 矩阵 (缩放系数 = 4)
static const ap_uint<5> BLEND_2x2_H[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 4, 4, 4, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

static const ap_uint<5> BLEND_2x2_V[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 4, 0, 0},
    {0, 0, 4, 0, 0},
    {0, 0, 4, 0, 0},
    {0, 0, 0, 0, 0}
};

static const ap_uint<5> BLEND_3x3[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 4, 4, 4, 0},
    {0, 4, 4, 4, 0},
    {0, 4, 4, 4, 0},
    {0, 0, 0, 0, 0}
};

static const ap_uint<5> BLEND_4x4[5][5] = {
    {4, 8, 8, 8, 4},
    {4, 16, 16, 16, 8},
    {4, 16, 16, 16, 8},
    {4, 16, 16, 16, 8},
    {4, 8, 8, 8, 4}
};

static const ap_uint<5> BLEND_5x5[5][5] = {
    {4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4}
};

static const ap_uint<5> BLEND_ZEROS[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

pixel_t iir_blend(
    pixel_t current,
    pixel_t prev,
    winsize_t win_size,
    ap_uint<8> blend_ratio[4])
{
    // 根据 win_size 选择融合系数
    // win_size: 16->idx=0, 24->idx=1, 32->idx=2, 40->idx=3
    ap_uint<2> idx = (ap_uint<2>)(win_size - 2);  // 2,3,4,5 -> 0,1,2,3

    ap_uint<8> ratio = blend_ratio[idx];

    // IIR blend: ratio * current + (64 - ratio) * prev / 64
    acc_t temp = (acc_t)ratio * (acc_t)current + (acc_t)(64 - ratio) * (acc_t)prev;

    // 除以 64 (带舍入)
    pixel_t result = (pixel_t)((temp + 32) / 64);

    return result;
}

void apply_blend_factor(
    pixel_t iir_avg,
    ap_uint<5> factor,
    pixel_t src,
    pixel_t &blend_out)
{
    // blend = (iir_avg * factor + src * (16 - factor)) / 16
    acc_t temp = (acc_t)iir_avg * (acc_t)factor + (acc_t)src * (acc_t)(16 - factor);

    // 除以 16 (带舍入)
    blend_out = (pixel_t)((temp + 8) / 16);
}

pixel_t final_blend(
    pixel_t blend0,
    pixel_t blend1,
    winsize_t win_size)
{
    // 计算权重
    // idx = win_size - 2 (2,3,4,5 -> 0,1,2,3)
    ap_uint<2> idx = (ap_uint<2>)(win_size - 2);

    // blend0_weight = 2 * idx + 1 = 1, 3, 5, 7
    // blend1_weight = 7 - 2 * idx = 7, 5, 3, 1
    ap_uint<4> blend0_weight = 2 * idx + 1;
    ap_uint<4> blend1_weight = 7 - 2 * idx;

    acc_t temp = (acc_t)blend0 * (acc_t)blend0_weight + (acc_t)blend1 * (acc_t)blend1_weight;

    // 除以 8 (带舍入)
    pixel_t result = (pixel_t)((temp + 4) / 8);

    return result;
}

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
    index_t width)
{
    // 上一行数据缓存
    pixel_t prev_line[MAX_IMAGE_WIDTH];

#pragma HLS ARRAY_PARTITION variable=prev_line cyclic factor=4

    // 初始化
    for (int i = 0; i < MAX_IMAGE_WIDTH; i++) {
        prev_line[i] = 0;
    }

    // 处理每个像素
    for (index_t row = 0; row < MAX_IMAGE_HEIGHT; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1

            // 读取输入
            pixel_t blend0_avg = blend0_avg_in.read();
            pixel_t blend1_avg = blend1_avg_in.read();
            pixel_t avg0_u = avg0_u_in.read();
            pixel_t avg1_u = avg1_u_in.read();
            winsize_t win_size = win_size_in.read();
            pixel_t pixel = pixel_in.read();
            ap_uint<1> last = last_in.read();

            // IIR Blending
            pixel_t blend0_iir = iir_blend(blend0_avg, avg0_u, win_size, blend_ratio);
            pixel_t blend1_iir = iir_blend(blend1_avg, avg1_u, win_size, blend_ratio);

            // 选择 Blend Factor (简化版: 使用中心点的 factor)
            ap_uint<5> factor0, factor1;

            switch (win_size) {
                case WINSIZE_2x2:
                    factor0 = 0;
                    factor1 = 4;
                    break;
                case WINSIZE_3x3:
                    factor0 = 4;
                    factor1 = 4;
                    break;
                case WINSIZE_4x4:
                    factor0 = 16;
                    factor1 = 4;
                    break;
                case WINSIZE_5x5:
                default:
                    factor0 = 4;
                    factor1 = 0;
                    break;
            }

            // 应用 Blend Factor
            pixel_t blend0_out, blend1_out;
            apply_blend_factor(blend0_iir, factor0, pixel, blend0_out);
            apply_blend_factor(blend1_iir, factor1, pixel, blend1_out);

            // 最终融合
            pixel_t output = final_blend(blend0_out, blend1_out, win_size);

            // 输出
            output_out.write(output);
            last_out.write(last);

            // 存储当前行数据
            prev_line[col] = output;

            // 行结束处理
            if (last) {
                break;
            }
        }
    }
}