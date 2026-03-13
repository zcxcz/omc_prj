/**
 * @file tb_csiir.cpp
 * @brief CSIIR 模块测试平台
 *
 * 功能验证测试平台
 *
 * @version 2.0
 * @date 2026-03-13
 *
 * 更新:
 * - 支持 10-bit 像素测试
 * - 支持大分辨率测试
 * - 支持 YUV 三通道测试
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "csiir_top.h"

// 测试图像尺寸 (可配置)
#define TEST_WIDTH  64
#define TEST_HEIGHT 64

// 大分辨率测试开关
// #define TEST_8K

#ifdef TEST_8K
#undef TEST_WIDTH
#undef TEST_HEIGHT
#define TEST_WIDTH  7680
#define TEST_HEIGHT 4320
#endif

// 生成测试输入 (支持参数化位宽)
void generate_test_input(hls::stream<AxisYUV> &axis_in, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            AxisYUV data;

            // Y: 渐变模式 (使用参数化最大值)
            data.y = (pixel_t)((col * PIXEL_MAX) / width);

            // U: 梯度模式
            data.u = (pixel_t)((row * PIXEL_MAX) / height);

            // V: 棋盘格模式
            data.v = ((row / 8 + col / 8) % 2 == 0) ? (pixel_t)(PIXEL_MAX * 80 / 100) : (pixel_t)(PIXEL_MAX * 20 / 100);

            data.user = (row == 0 && col == 0) ? 1 : 0;
            data.last = (col == width - 1) ? 1 : 0;

            axis_in.write(data);
        }
    }
}

// 验证输出 (支持参数化位宽)
bool verify_output(hls::stream<AxisYUV> &axis_out, int width, int height) {
    bool pass = true;
    int pixel_count = 0;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (axis_out.empty()) {
                std::cout << "Error: Output stream empty at (" << row << ", " << col << ")" << std::endl;
                return false;
            }

            AxisYUV data = axis_out.read();
            pixel_count++;

            // 基本范围检查 (使用参数化最大值)
            if (data.y > PIXEL_MAX || data.u > PIXEL_MAX || data.v > PIXEL_MAX) {
                std::cout << "Error: Output out of range at (" << row << ", " << col << "): "
                          << "Y=" << (int)data.y << ", U=" << (int)data.u << ", V=" << (int)data.v
                          << " (max=" << PIXEL_MAX << ")" << std::endl;
                pass = false;
            }
        }
    }

    std::cout << "Total pixels output: " << pixel_count << std::endl;
    return pass;
}

// 打印配置信息
void print_config_info() {
    std::cout << "\n=== CSIIR Configuration ===" << std::endl;
    std::cout << "Pixel Bitwidth: " << PIXEL_BITWIDTH << "-bit" << std::endl;
    std::cout << "Pixel Max Value: " << PIXEL_MAX << std::endl;
    std::cout << "Max Resolution: " << MAX_IMAGE_WIDTH << " x " << MAX_IMAGE_HEIGHT << std::endl;
    std::cout << "YUV Format: " << YUV_FORMAT << std::endl;
#if YUV_FORMAT == 444
    std::cout << "YUV Channels: 3 (YUV444, no subsampling)" << std::endl;
#elif YUV_FORMAT == 422
    std::cout << "YUV Channels: 2 (YUV422, horizontal 2:1 subsampling)" << std::endl;
#elif YUV_FORMAT == 420
    std::cout << "YUV Channels: 2 (YUV420, 2:1 subsampling)" << std::endl;
#endif
    std::cout << "Index Bitwidth: " << INDEX_BITWIDTH << "-bit" << std::endl;
    std::cout << "===========================" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "CSIIR Module Testbench v2.0" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 打印配置信息
    print_config_info();

    // 创建流
    hls::stream<AxisYUV> axis_in("axis_in");
    hls::stream<AxisYUV> axis_out("axis_out");

    // 配置 (阈值需要根据位宽调整)
    CSIIRConfig config;

    // 10-bit 像素的阈值 (范围 0-1023)
    // 原始 8-bit 阈值乘以 4 得到大致等价的 10-bit 阈值
    config.sobel_thresh_0 = 64;   // 2x2 窗口阈值 (原 16 * 4)
    config.sobel_thresh_1 = 96;   // 3x3 窗口阈值 (原 24 * 4)
    config.sobel_thresh_2 = 128;  // 4x4 窗口阈值 (原 32 * 4)
    config.sobel_thresh_3 = 160;  // 5x5 窗口阈值 (原 40 * 4)

    config.blend_coeff[0] = 32;
    config.blend_coeff[1] = 32;
    config.blend_coeff[2] = 32;
    config.blend_coeff[3] = 32;

    std::cout << "\nTest configuration:" << std::endl;
    std::cout << "  Image size: " << TEST_WIDTH << " x " << TEST_HEIGHT << std::endl;
    std::cout << "  Thresholds: " << (int)config.sobel_thresh_0 << ", "
              << (int)config.sobel_thresh_1 << ", "
              << (int)config.sobel_thresh_2 << ", "
              << (int)config.sobel_thresh_3 << std::endl;

    // 生成测试输入
    std::cout << "\nGenerating test input..." << std::endl;
    generate_test_input(axis_in, TEST_WIDTH, TEST_HEIGHT);
    std::cout << "Input pixels: " << TEST_WIDTH * TEST_HEIGHT << std::endl;

    // 运行 CSIIR
    std::cout << "\nRunning CSIIR (3-channel independent processing)..." << std::endl;
    csiir_top(axis_in, axis_out, config, TEST_WIDTH, TEST_HEIGHT);

    // 验证输出
    std::cout << "\nVerifying output..." << std::endl;
    bool pass = verify_output(axis_out, TEST_WIDTH, TEST_HEIGHT);

    // 结果
    std::cout << "\n============================================================" << std::endl;
    if (pass) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return pass ? 0 : 1;
}