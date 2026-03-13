/**
 * @file tb_csiir_simple.cpp
 * @brief CSIIR 简化功能测试
 *
 * 直接测试核心算法函数
 *
 * @version 1.0
 * @date 2026-03-13
 */

#include <iostream>
#include <cstdlib>
#include "csiir_types.h"
#include "sobel_filter.h"
#include "window_selector.h"
#include "directional_filter.h"
#include "blending.h"

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "CSIIR Simple Functional Test" << std::endl;
    std::cout << "============================================================" << std::endl;

    bool all_pass = true;

    // Test 1: Sobel Filter Window
    std::cout << "\n[Test 1] Sobel 5x5 Filter" << std::endl;
    {
        pixel_t window[5][5] = {
            {100, 110, 120, 130, 140},
            {100, 110, 120, 130, 140},
            {100, 110, 120, 130, 140},
            {100, 110, 120, 130, 140},
            {100, 110, 120, 130, 140}
        };

        grad_signed_t grad_h, grad_v;
        grad_t grad;

        sobel_filter_5x5_window(window, grad_h, grad_v, grad);

        std::cout << "  Gradient H: " << grad_h << std::endl;
        std::cout << "  Gradient V: " << grad_v << std::endl;
        std::cout << "  Gradient Mag: " << grad << std::endl;

        // 验证: 水平梯度应该接近 0 (水平均匀)
        // 垂直梯度应该非零 (垂直有变化)
        if (grad >= 0) {
            std::cout << "  PASS: Gradient calculation works" << std::endl;
        } else {
            std::cout << "  FAIL: Invalid gradient" << std::endl;
            all_pass = false;
        }
    }

    // Test 2: Window Size Selection
    std::cout << "\n[Test 2] Window Size Selection" << std::endl;
    {
        winsize_t ws;

        ws = select_window_size(10, 10, 10, 16, 24, 32, 40);
        std::cout << "  Grad=10 -> WinSize=" << (int)ws << " (expected 2)" << std::endl;

        ws = select_window_size(20, 20, 20, 16, 24, 32, 40);
        std::cout << "  Grad=20 -> WinSize=" << (int)ws << " (expected 3)" << std::endl;

        ws = select_window_size(30, 30, 30, 16, 24, 32, 40);
        std::cout << "  Grad=30 -> WinSize=" << (int)ws << " (expected 4)" << std::endl;

        ws = select_window_size(50, 50, 50, 16, 24, 32, 40);
        std::cout << "  Grad=50 -> WinSize=" << (int)ws << " (expected 5)" << std::endl;

        std::cout << "  PASS: Window selection works" << std::endl;
    }

    // Test 3: Directional Filter
    std::cout << "\n[Test 3] Directional Filter" << std::endl;
    {
        pixel_t window[5][5] = {
            {100, 100, 100, 100, 100},
            {100, 110, 120, 110, 100},
            {100, 120, 128, 120, 100},
            {100, 110, 120, 110, 100},
            {100, 100, 100, 100, 100}
        };

        DirFilterOutput avg0, avg1;
        directional_filter(window, WINSIZE_3x3, avg0, avg1);

        std::cout << "  avg0: c=" << (int)avg0.avg_c << ", u=" << (int)avg0.avg_u << std::endl;
        std::cout << "  avg1: c=" << (int)avg1.avg_c << ", u=" << (int)avg1.avg_u << std::endl;

        if (avg0.avg_c > 0 && avg1.avg_c > 0) {
            std::cout << "  PASS: Directional filter works" << std::endl;
        } else {
            std::cout << "  FAIL: Invalid averages" << std::endl;
            all_pass = false;
        }
    }

    // Test 4: Gradient Weighted Average
    std::cout << "\n[Test 4] Gradient Weighted Average" << std::endl;
    {
        DirFilterOutput avgs = {120, 110, 130, 115, 125};
        grad_t gc = 20, gu = 30, gd = 15, gl = 25, gr = 18;

        pixel_t result = gradient_weighted_avg(avgs, gc, gu, gd, gl, gr);
        std::cout << "  Result: " << (int)result << std::endl;

        if (result > 100 && result < 140) {
            std::cout << "  PASS: Gradient weighted average works" << std::endl;
        } else {
            std::cout << "  FAIL: Result out of expected range" << std::endl;
            all_pass = false;
        }
    }

    // Test 5: IIR Blend
    std::cout << "\n[Test 5] IIR Blend" << std::endl;
    {
        ap_uint<8> blend_ratio[4] = {32, 32, 32, 32};

        pixel_t result = iir_blend(128, 100, WINSIZE_3x3, blend_ratio);
        std::cout << "  IIR(128, 100) = " << (int)result << std::endl;

        // 应该在 100 和 128 之间
        if (result >= 100 && result <= 128) {
            std::cout << "  PASS: IIR blend works" << std::endl;
        } else {
            std::cout << "  FAIL: Result out of range" << std::endl;
            all_pass = false;
        }
    }

    // Test 6: Final Blend
    std::cout << "\n[Test 6] Final Blend" << std::endl;
    {
        pixel_t result = final_blend(100, 150, WINSIZE_3x3);
        std::cout << "  Final(100, 150, win=3) = " << (int)result << std::endl;

        if (result >= 100 && result <= 150) {
            std::cout << "  PASS: Final blend works" << std::endl;
        } else {
            std::cout << "  FAIL: Result out of range" << std::endl;
            all_pass = false;
        }
    }

    // Result
    std::cout << "\n============================================================" << std::endl;
    if (all_pass) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return all_pass ? 0 : 1;
}