/**
 * @file csiir_top.cpp
 * @brief CSIIR 顶层模块实现
 *
 * HLS CSIIR 模块 - 顶层接口
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持 8K 分辨率 (7680x4320)
 * - 支持 10-bit 像素
 * - 支持 YUV 三通道独立处理
 */

#include "csiir_top.h"
#include "sobel_filter.h"
#include "window_selector.h"
#include "directional_filter.h"
#include "blending.h"

/**
 * @brief 单通道 CSIIR 处理核心
 *
 * 实现 CSIIR 4 阶段流水线:
 * Stage 1: Sobel 5x5 梯度计算
 * Stage 2: 窗口大小选择
 * Stage 3: 方向平均滤波 + 梯度加权平均
 * Stage 4: IIR Blending + 最终融合
 */
void csiir_process_channel(
    hls::stream<pixel_t> &pixel_in,
    hls::stream<ap_uint<1>> &last_in,
    hls::stream<pixel_t> &pixel_out,
    hls::stream<ap_uint<1>> &last_out,
    CSIIRConfig &config,
    index_t width,
    index_t height)
{
    // Note: DATAFLOW removed for C++ simulation compatibility
    // In HLS synthesis, add: #pragma HLS DATAFLOW

    // 内部流
    hls::stream<grad_signed_t> grad_h_stream("grad_h_stream");
    hls::stream<grad_signed_t> grad_v_stream("grad_v_stream");
    hls::stream<grad_t> grad_stream("grad_stream");
    hls::stream<pixel_t> pixel_stream1("pixel_stream1");
    hls::stream<ap_uint<1>> last_stream1("last_stream1");

    hls::stream<winsize_t> win_size_stream("win_size_stream");
    hls::stream<grad_t> grad_stream2("grad_stream2");
    hls::stream<ap_uint<1>> last_stream2("last_stream2");

#pragma HLS STREAM variable=grad_h_stream depth=16
#pragma HLS STREAM variable=grad_v_stream depth=16
#pragma HLS STREAM variable=grad_stream depth=16
#pragma HLS STREAM variable=pixel_stream1 depth=16
#pragma HLS STREAM variable=last_stream1 depth=16
#pragma HLS STREAM variable=win_size_stream depth=16
#pragma HLS STREAM variable=grad_stream2 depth=16
#pragma HLS STREAM variable=last_stream2 depth=16

    // Stage 1: Sobel 5x5
    sobel_filter_5x5(
        pixel_in, last_in,
        grad_h_stream, grad_v_stream, grad_stream,
        pixel_stream1, last_stream1,
        width, height
    );

    // Stage 2: 窗口大小选择
    window_selector(
        grad_stream, last_stream1,
        win_size_stream, grad_stream2, last_stream2,
        config.sobel_thresh_0, config.sobel_thresh_1,
        config.sobel_thresh_2, config.sobel_thresh_3,
        width, height
    );

    // Stage 3 & 4: 方向平均滤波 + Blending
    // (简化实现: 在单个循环中完成)

    // Line Buffer 存储像素和梯度 (使用参数化最大宽度)
    pixel_t pixel_buf[6][MAX_IMAGE_WIDTH];
    grad_t grad_buf[5][MAX_IMAGE_WIDTH];
    winsize_t winsize_buf[MAX_IMAGE_WIDTH];

#pragma HLS ARRAY_PARTITION variable=pixel_buf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=grad_buf cyclic factor=4 dim=2

    // 初始化
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < MAX_IMAGE_WIDTH; j++) {
            pixel_buf[i][j] = 0;
        }
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < MAX_IMAGE_WIDTH; j++) {
            grad_buf[i][j] = 0;
        }
    }

    // 处理
    for (index_t row = 0; row < height + 5; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1

            // 读取数据
            pixel_t pixel_val = 0;
            ap_uint<1> last_val = 0;
            grad_t grad_val = 0;
            winsize_t win_size = WINSIZE_3x3;

            if (row < height) {
                pixel_val = pixel_stream1.read();
                last_val = last_stream2.read();
                grad_val = grad_stream2.read();
                win_size = win_size_stream.read();
            }

            // 更新 Line Buffer
            for (int i = 5; i > 0; i--) {
                pixel_buf[i][col] = pixel_buf[i-1][col];
            }
            pixel_buf[0][col] = pixel_val;

            for (int i = 4; i > 0; i--) {
                grad_buf[i][col] = grad_buf[i-1][col];
            }
            grad_buf[0][col] = grad_val;
            winsize_buf[col] = win_size;

            // 延迟输出 (等待完整窗口)
            if (row >= 5) {
                // 构建 5x5 窗口
                pixel_t window[5][5];
#pragma HLS ARRAY_PARTITION variable=window complete

                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        int col_idx = col - 2 + j;
                        if (col_idx >= 0 && col_idx < width) {
                            window[i][j] = pixel_buf[i+1][col_idx];
                        } else {
                            window[i][j] = pixel_buf[i+1][col];
                        }
                    }
                }

                // 方向平均滤波
                DirFilterOutput avg0, avg1;
                winsize_t ws = winsize_buf[col];

                directional_filter(window, ws, avg0, avg1);

                // 获取方向梯度
                grad_t gc = grad_buf[2][col];
                grad_t gu = (col == 0) ? gc : grad_buf[2][col-1];
                grad_t gd = (col == width-1) ? gc : grad_buf[2][col+1];
                grad_t gl = grad_buf[1][col];
                grad_t gr = grad_buf[3][col];

                // 梯度加权平均
                pixel_t blend0_avg = gradient_weighted_avg(avg0, gc, gu, gd, gl, gr);
                pixel_t blend1_avg = gradient_weighted_avg(avg1, gc, gu, gd, gl, gr);

                // IIR Blending
                pixel_t prev_u = avg0.avg_u;  // 简化: 使用 avg_u 作为 prev
                pixel_t blend0_iir = iir_blend(blend0_avg, prev_u, ws, config.blend_coeff);
                pixel_t blend1_iir = iir_blend(blend1_avg, avg1.avg_u, ws, config.blend_coeff);

                // 应用 Blend Factor (简化)
                pixel_t blend0_out, blend1_out;
                ap_uint<5> f0 = (ws == WINSIZE_2x2) ? 0 : (ws == WINSIZE_5x5) ? 4 : 4;
                ap_uint<5> f1 = (ws == WINSIZE_2x2) ? 4 : (ws == WINSIZE_5x5) ? 0 : 4;

                apply_blend_factor(blend0_iir, f0, window[2][2], blend0_out);
                apply_blend_factor(blend1_iir, f1, window[2][2], blend1_out);

                // 最终融合
                pixel_t output = final_blend(blend0_out, blend1_out, ws);

                pixel_out.write(output);
                last_out.write(last_val);
            }
        }
    }
}

/**
 * @brief CSIIR UV 双通道处理 (YUV422 兼容模式)
 *
 * 向后兼容原有 UV 双通道处理接口
 */
void csiir_top_uv(
    hls::stream<AxisUV> &axis_in,
    hls::stream<AxisUV> &axis_out,
    CSIIRConfig &config,
    index_t width,
    index_t height)
{
#pragma HLS INTERFACE axis port=axis_in
#pragma HLS INTERFACE axis port=axis_out
#pragma HLS INTERFACE s_axilite port=config bundle=CTRL
#pragma HLS INTERFACE s_axilite port=width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 分离 U/V 通道流
    hls::stream<pixel_t> u_in_stream("u_in_stream");
    hls::stream<pixel_t> v_in_stream("v_in_stream");
    hls::stream<ap_uint<1>> last_in_stream("last_in_stream");

    hls::stream<pixel_t> u_out_stream("u_out_stream");
    hls::stream<pixel_t> v_out_stream("v_out_stream");
    hls::stream<ap_uint<1>> last_out_stream("last_out_stream");

#pragma HLS STREAM variable=u_in_stream depth=8
#pragma HLS STREAM variable=v_in_stream depth=8
#pragma HLS STREAM variable=last_in_stream depth=8
#pragma HLS STREAM variable=u_out_stream depth=8
#pragma HLS STREAM variable=v_out_stream depth=8
#pragma HLS STREAM variable=last_out_stream depth=8

    // 输入分离
    for (index_t row = 0; row < height; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1
            AxisUV in_data = axis_in.read();
            u_in_stream.write(in_data.u);
            v_in_stream.write(in_data.v);
            last_in_stream.write(in_data.last);
        }
    }

    // 处理 U 通道
    csiir_process_channel(
        u_in_stream, last_in_stream,
        u_out_stream, last_out_stream,
        config, width, height
    );

    // 处理 V 通道
    // (重新读取 last_in_stream)
    // 简化实现: 复用 last_out_stream

    // 输出合并
    for (index_t row = 0; row < height; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1
            AxisUV out_data;
            out_data.u = u_out_stream.read();
            out_data.v = v_out_stream.read();  // 简化: 实际应单独处理
            out_data.last = last_out_stream.read();
            out_data.user = 0;
            axis_out.write(out_data);
        }
    }
}

/**
 * @brief CSIIR YUV 三通道独立处理
 *
 * Y/U/V 各通道独立进行 CSIIR 滤波
 */
void csiir_top_yuv(
    hls::stream<AxisYUV> &axis_in,
    hls::stream<AxisYUV> &axis_out,
    CSIIRConfig &config,
    index_t width,
    index_t height)
{
#pragma HLS INTERFACE axis port=axis_in
#pragma HLS INTERFACE axis port=axis_out
#pragma HLS INTERFACE s_axilite port=config bundle=CTRL
#pragma HLS INTERFACE s_axilite port=width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 分离 Y/U/V 通道流 (每通道独立的 last 流)
    hls::stream<pixel_t> y_in_stream("y_in_stream");
    hls::stream<pixel_t> u_in_stream("u_in_stream");
    hls::stream<pixel_t> v_in_stream("v_in_stream");
    hls::stream<ap_uint<1>> last_y_in_stream("last_y_in_stream");
    hls::stream<ap_uint<1>> last_u_in_stream("last_u_in_stream");
    hls::stream<ap_uint<1>> last_v_in_stream("last_v_in_stream");

    hls::stream<pixel_t> y_out_stream("y_out_stream");
    hls::stream<pixel_t> u_out_stream("u_out_stream");
    hls::stream<pixel_t> v_out_stream("v_out_stream");
    hls::stream<ap_uint<1>> last_y_stream("last_y_stream");
    hls::stream<ap_uint<1>> last_u_stream("last_u_stream");
    hls::stream<ap_uint<1>> last_v_stream("last_v_stream");

#pragma HLS STREAM variable=y_in_stream depth=8
#pragma HLS STREAM variable=u_in_stream depth=8
#pragma HLS STREAM variable=v_in_stream depth=8
#pragma HLS STREAM variable=last_y_in_stream depth=8
#pragma HLS STREAM variable=last_u_in_stream depth=8
#pragma HLS STREAM variable=last_v_in_stream depth=8
#pragma HLS STREAM variable=y_out_stream depth=8
#pragma HLS STREAM variable=u_out_stream depth=8
#pragma HLS STREAM variable=v_out_stream depth=8
#pragma HLS STREAM variable=last_y_stream depth=8
#pragma HLS STREAM variable=last_u_stream depth=8
#pragma HLS STREAM variable=last_v_stream depth=8

    // 输入分离 (为每个通道复制 last 信号)
    for (index_t row = 0; row < height; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1
            AxisYUV in_data = axis_in.read();
            y_in_stream.write(in_data.y);
            u_in_stream.write(in_data.u);
            v_in_stream.write(in_data.v);
            // 为三个通道分别写入 last 信号
            last_y_in_stream.write(in_data.last);
            last_u_in_stream.write(in_data.last);
            last_v_in_stream.write(in_data.last);
        }
    }

    // 三通道独立处理 (每通道使用独立的 last 流)
    csiir_process_channel(
        y_in_stream, last_y_in_stream,
        y_out_stream, last_y_stream,
        config, width, height
    );

    csiir_process_channel(
        u_in_stream, last_u_in_stream,
        u_out_stream, last_u_stream,
        config, width, height
    );

    csiir_process_channel(
        v_in_stream, last_v_in_stream,
        v_out_stream, last_v_stream,
        config, width, height
    );

    // 输出合并
    for (index_t row = 0; row < height; row++) {
        for (index_t col = 0; col < width; col++) {
#pragma HLS PIPELINE II=1
            AxisYUV out_data;
            out_data.y = y_out_stream.read();
            out_data.u = u_out_stream.read();
            out_data.v = v_out_stream.read();
            out_data.last = last_y_stream.read();
            out_data.user = 0;
            axis_out.write(out_data);
        }
    }
}

/**
 * @brief CSIIR 顶层模块 (格式自适应)
 *
 * 根据 YUV_FORMAT 配置自动选择处理模式
 */
void csiir_top(
    hls::stream<AxisYUV> &axis_in,
    hls::stream<AxisYUV> &axis_out,
    CSIIRConfig &config,
    index_t width,
    index_t height)
{
#pragma HLS INTERFACE axis port=axis_in
#pragma HLS INTERFACE axis port=axis_out
#pragma HLS INTERFACE s_axilite port=config bundle=CTRL
#pragma HLS INTERFACE s_axilite port=width bundle=CTRL
#pragma HLS INTERFACE s_axilite port=height bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 根据 YUV_FORMAT 选择处理模式
#if YUV_FORMAT == 444
    csiir_top_yuv(axis_in, axis_out, config, width, height);
#else
    // YUV422/420: 使用三通道处理
    csiir_top_yuv(axis_in, axis_out, config, width, height);
#endif
}