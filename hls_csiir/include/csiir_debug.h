/**
 * @file csiir_debug.h
 * @brief CSIIR Debug Configuration
 *
 * Debug switches for intermediate output generation.
 * Enable at compile time with -DDEBUG_*=1 flags.
 *
 * @version 1.0
 * @date 2026-03-15
 */

#ifndef CSIIR_DEBUG_H
#define CSIIR_DEBUG_H

//=============================================================================
// Debug Switches (compile-time via -D flag)
//=============================================================================

/**
 * @brief Debug output for Sobel filter stage
 * Outputs: Row, Col, Grad_H, Grad_V, Grad_Mag
 */
#ifndef DEBUG_SOBEL
#define DEBUG_SOBEL 0
#endif

/**
 * @brief Debug output for Window Selector stage
 * Outputs: Row, Col, WinSize, Gradient
 */
#ifndef DEBUG_WINDOW
#define DEBUG_WINDOW 0
#endif

/**
 * @brief Debug output for Directional Filter stage
 * Outputs: Row, Col, Avg0, Avg1, WinSize
 */
#ifndef DEBUG_DIRECTIONAL
#define DEBUG_DIRECTIONAL 0
#endif

/**
 * @brief Debug output for Blending stage
 * Outputs: Row, Col, Blend0, Blend1, Final
 */
#ifndef DEBUG_BLENDING
#define DEBUG_BLENDING 0
#endif

/**
 * @brief Enable all debug outputs
 */
#ifndef DEBUG_ALL
#define DEBUG_ALL 0
#endif

//=============================================================================
// Derived Debug Flags
//=============================================================================

#if DEBUG_ALL
    #define DEBUG_OUTPUT_ENABLE 1
    #define DEBUG_SOBEL_ENABLE 1
    #define DEBUG_WINDOW_ENABLE 1
    #define DEBUG_DIRECTIONAL_ENABLE 1
    #define DEBUG_BLENDING_ENABLE 1
#else
    #define DEBUG_OUTPUT_ENABLE (DEBUG_SOBEL || DEBUG_WINDOW || DEBUG_DIRECTIONAL || DEBUG_BLENDING)
    #define DEBUG_SOBEL_ENABLE DEBUG_SOBEL
    #define DEBUG_WINDOW_ENABLE DEBUG_WINDOW
    #define DEBUG_DIRECTIONAL_ENABLE DEBUG_DIRECTIONAL
    #define DEBUG_BLENDING_ENABLE DEBUG_BLENDING
#endif

//=============================================================================
// Debug Output Helpers
//=============================================================================

#include <fstream>
#include <sstream>
#include <string>

namespace csiir_debug {

/**
 * @brief Debug file manager for intermediate output
 */
class DebugLogger {
public:
    static DebugLogger& instance() {
        static DebugLogger inst;
        return inst;
    }

    void set_output_dir(const std::string& dir) {
        output_dir_ = dir;
    }

    void log_sobel(int row, int col, int grad_h, int grad_v, int grad_mag) {
#if DEBUG_SOBEL_ENABLE
        if (!sobel_file_.is_open()) {
            sobel_file_.open(output_dir_ + "/debug_sobel.txt");
            sobel_file_ << "# Row, Col, Grad_H, Grad_V, Grad_Mag\n";
        }
        sobel_file_ << row << ", " << col << ", " << grad_h << ", " << grad_v << ", " << grad_mag << "\n";
#endif
    }

    void log_window(int row, int col, int win_size, int grad) {
#if DEBUG_WINDOW_ENABLE
        if (!window_file_.is_open()) {
            window_file_.open(output_dir_ + "/debug_window.txt");
            window_file_ << "# Row, Col, WinSize, Gradient\n";
        }
        window_file_ << row << ", " << col << ", " << win_size << ", " << grad << "\n";
#endif
    }

    void log_directional(int row, int col, int avg0, int avg1, int win_size) {
#if DEBUG_DIRECTIONAL_ENABLE
        if (!directional_file_.is_open()) {
            directional_file_.open(output_dir_ + "/debug_directional.txt");
            directional_file_ << "# Row, Col, Avg0, Avg1, WinSize\n";
        }
        directional_file_ << row << ", " << col << ", " << avg0 << ", " << avg1 << ", " << win_size << "\n";
#endif
    }

    void log_blending(int row, int col, int blend0, int blend1, int final_val) {
#if DEBUG_BLENDING_ENABLE
        if (!blending_file_.is_open()) {
            blending_file_.open(output_dir_ + "/debug_blending.txt");
            blending_file_ << "# Row, Col, Blend0, Blend1, Final\n";
        }
        blending_file_ << row << ", " << col << ", " << blend0 << ", " << blend1 << ", " << final_val << "\n";
#endif
    }

    void close_all() {
#if DEBUG_SOBEL_ENABLE
        if (sobel_file_.is_open()) sobel_file_.close();
#endif
#if DEBUG_WINDOW_ENABLE
        if (window_file_.is_open()) window_file_.close();
#endif
#if DEBUG_DIRECTIONAL_ENABLE
        if (directional_file_.is_open()) directional_file_.close();
#endif
#if DEBUG_BLENDING_ENABLE
        if (blending_file_.is_open()) blending_file_.close();
#endif
    }

private:
    DebugLogger() = default;
    ~DebugLogger() { close_all(); }
    DebugLogger(const DebugLogger&) = delete;
    DebugLogger& operator=(const DebugLogger&) = delete;

    std::string output_dir_ = ".";

#if DEBUG_SOBEL_ENABLE
    std::ofstream sobel_file_;
#endif
#if DEBUG_WINDOW_ENABLE
    std::ofstream window_file_;
#endif
#if DEBUG_DIRECTIONAL_ENABLE
    std::ofstream directional_file_;
#endif
#if DEBUG_BLENDING_ENABLE
    std::ofstream blending_file_;
#endif
};

} // namespace csiir_debug

#endif // CSIIR_DEBUG_H