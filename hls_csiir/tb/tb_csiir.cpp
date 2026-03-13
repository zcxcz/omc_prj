/**
 * @file tb_csiir.cpp
 * @brief CSIIR 模块测试平台
 *
 * 功能验证测试平台
 *
 * @version 1.0
 * @date 2026-03-13
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "csiir_top.h"

// 测试图像尺寸
#define TEST_WIDTH  64
#define TEST_HEIGHT 64

// 生成测试输入
void generate_test_input(hls::stream<AxisUV> &axis_in, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            AxisUV data;

            // U: 梯度模式
            data.u = (pixel_t)(col * 255 / width);

            // V: 棋盘格模式
            data.v = ((row / 8 + col / 8) % 2 == 0) ? (pixel_t)200 : (pixel_t)50;

            data.user = (row == 0 && col == 0) ? 1 : 0;
            data.last = (col == width - 1) ? 1 : 0;

            axis_in.write(data);
        }
    }
}

// 验证输出
bool verify_output(hls::stream<AxisUV> &axis_out, int width, int height) {
    bool pass = true;
    int pixel_count = 0;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (axis_out.empty()) {
                std::cout << "Error: Output stream empty at (" << row << ", " << col << ")" << std::endl;
                return false;
            }

            AxisUV data = axis_out.read();
            pixel_count++;

            // 基本范围检查
            if (data.u > 255 || data.v > 255) {
                std::cout << "Error: Output out of range at (" << row << ", " << col << "): "
                          << "U=" << (int)data.u << ", V=" << (int)data.v << std::endl;
                pass = false;
            }
        }
    }

    std::cout << "Total pixels output: " << pixel_count << std::endl;
    return pass;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "CSIIR Module Testbench" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 创建流
    hls::stream<AxisUV> axis_in("axis_in");
    hls::stream<AxisUV> axis_out("axis_out");

    // 配置
    CSIIRConfig config;
    config.sobel_thresh_0 = 16;
    config.sobel_thresh_1 = 24;
    config.sobel_thresh_2 = 32;
    config.sobel_thresh_3 = 40;
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
    std::cout << "\nRunning CSIIR..." << std::endl;
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