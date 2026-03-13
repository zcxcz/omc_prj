/**
 * @file window_selector.cpp
 * @brief 窗口大小选择模块实现
 *
 * HLS CSIIR 模块 - Stage 2: 窗口大小选择
 *
 * @version 1.0
 * @date 2026-03-13
 */

#include "window_selector.h"

winsize_t select_window_size(
    grad_t grad_curr,
    grad_t grad_prev,
    grad_t grad_next,
    ap_uint<8> thresh_0,
    ap_uint<8> thresh_1,
    ap_uint<8> thresh_2,
    ap_uint<8> thresh_3)
{
    // 使用 3 个连续像素的最大梯度
    grad_t max_grad = grad_curr;
    if (grad_prev > max_grad) max_grad = grad_prev;
    if (grad_next > max_grad) max_grad = grad_next;

    // 阈值比较选择窗口大小
    winsize_t win_size;

    if (max_grad < thresh_0) {
        win_size = WINSIZE_2x2;  // 2
    } else if (max_grad < thresh_1) {
        win_size = WINSIZE_3x3;  // 3
    } else if (max_grad < thresh_2) {
        win_size = WINSIZE_4x4;  // 4
    } else {
        win_size = WINSIZE_5x5;  // 5
    }

    return win_size;
}

void window_selector(
    hls::stream<grad_t> &grad_in,
    hls::stream<ap_uint<1>> &last_in,
    hls::stream<winsize_t> &win_size_out,
    hls::stream<grad_t> &grad_out,
    hls::stream<ap_uint<1>> &last_out,
    ap_uint<8> thresh_0,
    ap_uint<8> thresh_1,
    ap_uint<8> thresh_2,
    ap_uint<8> thresh_3,
    index_t width)
{
    // 行缓存 (存储上一行梯度)
    grad_t prev_line[MAX_IMAGE_WIDTH];
    grad_t curr_line[MAX_IMAGE_WIDTH];

#pragma HLS ARRAY_PARTITION variable=prev_line cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=curr_line cyclic factor=4

    // 初始化
    for (int i = 0; i < MAX_IMAGE_WIDTH; i++) {
#pragma HLS UNROLL
        prev_line[i] = 0;
        curr_line[i] = 0;
    }

    // 当前像素的前一列梯度
    grad_t grad_prev_col = 0;

    // 处理每个像素
    for (index_t row = 0; row < MAX_IMAGE_HEIGHT; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1

            // 读取输入
            grad_t grad_curr = grad_in.read();
            ap_uint<1> last = last_in.read();

            // 获取前一列和后一列的梯度
            grad_t grad_prev = (col == 0) ? grad_curr : grad_prev_col;
            grad_t grad_next = (col == width - 1) ? grad_curr : curr_line[col + 1];

            // 更新缓存
            grad_prev_col = grad_curr;

            // 存储当前行梯度 (用于下一行)
            curr_line[col] = grad_curr;

            // 窗口大小选择
            winsize_t win_size = select_window_size(
                grad_curr, grad_prev, grad_next,
                thresh_0, thresh_1, thresh_2, thresh_3
            );

            // 输出
            win_size_out.write(win_size);
            grad_out.write(grad_curr);
            last_out.write(last);

            // 行结束时更新行缓存
            if (last) {
                // 交换行缓存
                for (int i = 0; i < width; i++) {
                    prev_line[i] = curr_line[i];
                    curr_line[i] = 0;
                }
                break;  // 退出当前行循环
            }
        }
    }
}