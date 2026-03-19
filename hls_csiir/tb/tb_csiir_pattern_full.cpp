/**
 * @file tb_csiir_pattern_full.cpp
 * @brief CSIIR Pattern Output Testbench - Full Pipeline Version
 *
 * Uses complete HLS pipeline modules and captures intermediate data
 * at each stage for comparison with Python reference.
 *
 * Pipeline:
 *   Stage 1: sobel_filter_5x5() - Sobel gradient computation
 *   Stage 2: window_selector() - Window size selection
 *   Stage 3: directional_filter() - Directional averaging
 *   Stage 4: blending_pipeline() - IIR blending and final fusion
 *
 * Usage: tb_csiir_pattern_full <input.bin> <output_dir> <width> <height>
 *
 * @version 2.0
 * @date 2026-03-18
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <errno.h>
#include "csiir_top.h"
#include "sobel_filter.h"
#include "window_selector.h"
#include "directional_filter.h"
#include "blending.h"

//=============================================================================
// Binary File Format
//=============================================================================

#pragma pack(push, 1)
struct BinaryHeader {
    char magic[4];
    uint16_t width;
    uint16_t height;
    uint8_t pixel_bits;
    uint8_t channels;
    uint8_t reserved[6];
};
#pragma pack(pop)

const char MAGIC[] = "CSII";
const size_t HEADER_SIZE = 16;

//=============================================================================
// Pattern Output Structure (per channel)
//=============================================================================

struct ChannelPatternData {
    // Stage 1: Sobel
    std::vector<int32_t> grad_h;
    std::vector<int32_t> grad_v;
    std::vector<uint32_t> grad_magnitude;

    // Stage 2: Window Selector
    std::vector<uint8_t> win_size;
    std::vector<uint32_t> grad_used;

    // Stage 3: Directional Filter
    std::vector<uint16_t> avg_c;
    std::vector<uint16_t> avg_u;
    std::vector<uint16_t> avg_d;
    std::vector<uint16_t> avg_l;
    std::vector<uint16_t> avg_r;
    std::vector<uint16_t> blend0_avg;
    std::vector<uint16_t> blend1_avg;

    // Stage 4: Blending
    std::vector<uint16_t> blend0_iir;
    std::vector<uint16_t> blend1_iir;
    std::vector<uint16_t> final_output;

    void resize(size_t n) {
        grad_h.resize(n, 0);
        grad_v.resize(n, 0);
        grad_magnitude.resize(n, 0);
        win_size.resize(n, 0);
        grad_used.resize(n, 0);
        avg_c.resize(n, 0);
        avg_u.resize(n, 0);
        avg_d.resize(n, 0);
        avg_l.resize(n, 0);
        avg_r.resize(n, 0);
        blend0_avg.resize(n, 0);
        blend1_avg.resize(n, 0);
        blend0_iir.resize(n, 0);
        blend1_iir.resize(n, 0);
        final_output.resize(n, 0);
    }
};

//=============================================================================
// NPY File Writer
//=============================================================================

bool write_npy_int32(const std::string& filepath, const int32_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<i4', 'fortran_order': False, 'shape': (" << height << ", " << width << "), }";
    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) header_dict += std::string(padding - 1, ' ') + '\n';
    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), (size_t)width * height * sizeof(int32_t));
    return true;
}

bool write_npy_uint32(const std::string& filepath, const uint32_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<u4', 'fortran_order': False, 'shape': (" << height << ", " << width << "), }";
    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) header_dict += std::string(padding - 1, ' ') + '\n';
    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), (size_t)width * height * sizeof(uint32_t));
    return true;
}

bool write_npy_uint16(const std::string& filepath, const uint16_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<u2', 'fortran_order': False, 'shape': (" << height << ", " << width << "), }";
    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) header_dict += std::string(padding - 1, ' ') + '\n';
    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), (size_t)width * height * sizeof(uint16_t));
    return true;
}

bool write_npy_uint8(const std::string& filepath, const uint8_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<u1', 'fortran_order': False, 'shape': (" << height << ", " << width << "), }";
    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) header_dict += std::string(padding - 1, ' ') + '\n';
    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), (size_t)width * height);
    return true;
}

//=============================================================================
// Directory Utilities
//=============================================================================

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

//=============================================================================
// Helper Functions
//=============================================================================

bool read_binary_input(const char* filepath,
                       std::vector<uint16_t>& y_data,
                       std::vector<uint16_t>& u_data,
                       std::vector<uint16_t>& v_data,
                       BinaryHeader& header) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    file.read(reinterpret_cast<char*>(&header), HEADER_SIZE);
    if (strncmp(header.magic, MAGIC, 4) != 0) return false;

    size_t total_pixels = (size_t)header.width * header.height;
    y_data.resize(total_pixels);
    u_data.resize(total_pixels);
    v_data.resize(total_pixels);

    if (header.pixel_bits <= 8) {
        std::vector<uint8_t> raw(total_pixels * 3);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size());
        for (size_t i = 0; i < total_pixels; i++) {
            y_data[i] = raw[i * 3 + 0];
            u_data[i] = raw[i * 3 + 1];
            v_data[i] = raw[i * 3 + 2];
        }
    } else {
        std::vector<uint16_t> raw(total_pixels * 3);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size() * 2);
        for (size_t i = 0; i < total_pixels; i++) {
            y_data[i] = raw[i * 3 + 0];
            u_data[i] = raw[i * 3 + 1];
            v_data[i] = raw[i * 3 + 2];
        }
    }
    return true;
}

//=============================================================================
// Full Pipeline Processing with Pattern Capture
//=============================================================================

void process_channel_full_pipeline(
    const std::vector<uint16_t>& channel_in,
    std::vector<uint16_t>& channel_out,
    ChannelPatternData& pattern,
    uint16_t width,
    uint16_t height,
    uint16_t pixel_max)
{
    size_t total_pixels = (size_t)width * height;
    channel_out.resize(total_pixels);
    pattern.resize(total_pixels);

    // Configuration (scaled for pixel bit depth)
    ap_uint<16> thresh[4] = {64, 96, 128, 160};  // 10-bit thresholds
    ap_uint<8> blend_ratio[4] = {32, 32, 32, 32};

    // Create HLS streams
    hls::stream<pixel_t> pixel_in("pixel_in");
    hls::stream<ap_uint<1>> last_in("last_in");

    // Stage 1 outputs
    hls::stream<grad_signed_t> grad_h_stream("grad_h_stream");
    hls::stream<grad_signed_t> grad_v_stream("grad_v_stream");
    hls::stream<grad_t> grad_stream("grad_stream");
    hls::stream<pixel_t> pixel_s1_out("pixel_s1_out");
    hls::stream<ap_uint<1>> last_s1_out("last_s1_out");

    // Stage 2 outputs
    hls::stream<winsize_t> win_size_stream("win_size_stream");
    hls::stream<grad_t> grad_s2_out("grad_s2_out");
    hls::stream<ap_uint<1>> last_s2_out("last_s2_out");

    // Write input to stream
    for (size_t i = 0; i < total_pixels; i++) {
        pixel_in.write((pixel_t)channel_in[i]);
        last_in.write((i % width) == (size_t)(width - 1) ? 1 : 0);
    }

    // Run Stage 1: Sobel filter
    sobel_filter_5x5(pixel_in, last_in, grad_h_stream, grad_v_stream, grad_stream,
                     pixel_s1_out, last_s1_out, (index_t)width, (index_t)height);

    // Capture Stage 1 outputs and prepare for Stage 2
    std::vector<grad_t> grad_map(total_pixels);
    std::vector<pixel_t> pixel_buf(total_pixels);

    for (size_t i = 0; i < total_pixels; i++) {
        grad_signed_t gh = grad_h_stream.read();
        grad_signed_t gv = grad_v_stream.read();
        grad_t g = grad_stream.read();
        pixel_t p = pixel_s1_out.read();
        ap_uint<1> last = last_s1_out.read();

        pattern.grad_h[i] = (int32_t)gh;
        pattern.grad_v[i] = (int32_t)gv;
        pattern.grad_magnitude[i] = (uint32_t)g;
        grad_map[i] = g;
        pixel_buf[i] = p;

        // Re-feed for Stage 2
        grad_stream.write(g);
        last_s2_out.write(last);
    }

    // Run Stage 2: Window selector
    window_selector(grad_stream, last_s2_out, win_size_stream, grad_s2_out, last_s2_out,
                    thresh[0], thresh[1], thresh[2], thresh[3], (index_t)width, (index_t)height);

    // Capture Stage 2 outputs
    std::vector<winsize_t> win_size_buf(total_pixels);
    for (size_t i = 0; i < total_pixels; i++) {
        win_size_buf[i] = win_size_stream.read();
        grad_t g = grad_s2_out.read();
        last_s2_out.read();

        pattern.win_size[i] = (uint8_t)win_size_buf[i];
        pattern.grad_used[i] = (uint32_t)grad_map[i];  // Use original gradient
    }

    // Process Stages 3 & 4 pixel by pixel (with line buffers)
    // This matches the HLS implementation in csiir_top.cpp

    // Line buffers for 5x5 window
    pixel_t line_buf[6][MAX_IMAGE_WIDTH];
    grad_t grad_buf[5][MAX_IMAGE_WIDTH];
    winsize_t winsize_buf[MAX_IMAGE_WIDTH];

    // Initialize buffers with reflect padding (use first row/col values)
    pixel_t first_pixel = pixel_buf[0];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < MAX_IMAGE_WIDTH; j++)
            line_buf[i][j] = first_pixel;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < MAX_IMAGE_WIDTH; j++)
            grad_buf[i][j] = 0;  // Gradient is 0 for flat image

    // Process with 5-line delay for 5x5 window
    for (size_t row = 0; row < (size_t)height + 5; row++) {
        for (size_t col = 0; col < (size_t)width; col++) {
            // Read new data
            pixel_t pixel_val = 0;
            grad_t grad_val = 0;
            winsize_t ws = 3;

            if (row < (size_t)height) {
                pixel_val = pixel_buf[row * width + col];
                grad_val = grad_map[row * width + col];
                ws = win_size_buf[row * width + col];
            }

            // Update line buffers (shift down)
            for (int i = 5; i > 0; i--) {
                line_buf[i][col] = line_buf[i-1][col];
            }
            line_buf[0][col] = pixel_val;

            for (int i = 4; i > 0; i--) {
                grad_buf[i][col] = grad_buf[i-1][col];
            }
            grad_buf[0][col] = grad_val;
            winsize_buf[col] = ws;

            // Output with 5-line delay
            if (row >= 5) {
                size_t out_idx = (row - 5) * width + col;
                winsize_t win_size = winsize_buf[col];

                // Build 5x5 window
                pixel_t window[5][5];
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        int c = (int)col - 2 + j;
                        if (c >= 0 && c < width) {
                            window[i][j] = line_buf[i+1][c];
                        } else {
                            window[i][j] = line_buf[i+1][col];
                        }
                    }
                }

                // Stage 3: Directional filter
                DirFilterOutput avg0, avg1;
                directional_filter(window, win_size, avg0, avg1);

                // Store Stage 3 outputs
                pattern.avg_c[out_idx] = (uint16_t)avg0.avg_c;
                pattern.avg_u[out_idx] = (uint16_t)avg0.avg_u;
                pattern.avg_d[out_idx] = (uint16_t)avg0.avg_d;
                pattern.avg_l[out_idx] = (uint16_t)avg0.avg_l;
                pattern.avg_r[out_idx] = (uint16_t)avg0.avg_r;

                // Get directional gradients
                grad_t gc = grad_buf[2][col];
                grad_t gu = grad_buf[1][col];  // Above
                grad_t gd = grad_buf[3][col];  // Below
                grad_t gl = (col == 0) ? gc : grad_buf[2][col-1];  // Left
                grad_t gr = (col == (size_t)width-1) ? gc : grad_buf[2][col+1];  // Right

                // Gradient weighted average
                pixel_t b0_avg = gradient_weighted_avg(avg0, gc, gu, gd, gl, gr);
                pixel_t b1_avg = gradient_weighted_avg(avg1, gc, gu, gd, gl, gr);

                pattern.blend0_avg[out_idx] = (uint16_t)b0_avg;
                pattern.blend1_avg[out_idx] = (uint16_t)b1_avg;

                // Stage 4: IIR blending
                pixel_t b0_iir = iir_blend(b0_avg, avg0.avg_u, win_size, blend_ratio);
                pixel_t b1_iir = iir_blend(b1_avg, avg1.avg_u, win_size, blend_ratio);

                pattern.blend0_iir[out_idx] = (uint16_t)b0_iir;
                pattern.blend1_iir[out_idx] = (uint16_t)b1_iir;

                // Apply blend factors
                pixel_t blend0_out, blend1_out;
                ap_uint<5> f0 = (win_size == 2) ? 0 : (win_size == 5) ? 4 : 4;
                ap_uint<5> f1 = (win_size == 2) ? 4 : (win_size == 5) ? 0 : 4;

                apply_blend_factor(b0_iir, f0, window[2][2], blend0_out);
                apply_blend_factor(b1_iir, f1, window[2][2], blend1_out);

                // Final blend
                pixel_t output = final_blend(blend0_out, blend1_out, win_size);

                pattern.final_output[out_idx] = (uint16_t)output;
                channel_out[out_idx] = (uint16_t)output;
            }
        }
    }
}

//=============================================================================
// Save Pattern Data
//=============================================================================

bool save_pattern_data(const std::string& base_dir, const std::string& channel,
                       const ChannelPatternData& pattern, uint16_t width, uint16_t height) {
    std::string ch_dir = base_dir + "/" + channel;
    create_directory(ch_dir);

    // Stage 1
    write_npy_int32(ch_dir + "/grad_h.npy", pattern.grad_h.data(), width, height);
    write_npy_int32(ch_dir + "/grad_v.npy", pattern.grad_v.data(), width, height);
    write_npy_uint32(ch_dir + "/grad_magnitude.npy", pattern.grad_magnitude.data(), width, height);

    // Stage 2
    write_npy_uint8(ch_dir + "/win_size.npy", pattern.win_size.data(), width, height);
    write_npy_uint32(ch_dir + "/grad_used.npy", pattern.grad_used.data(), width, height);

    // Stage 3
    write_npy_uint16(ch_dir + "/avg_c.npy", pattern.avg_c.data(), width, height);
    write_npy_uint16(ch_dir + "/avg_u.npy", pattern.avg_u.data(), width, height);
    write_npy_uint16(ch_dir + "/avg_d.npy", pattern.avg_d.data(), width, height);
    write_npy_uint16(ch_dir + "/avg_l.npy", pattern.avg_l.data(), width, height);
    write_npy_uint16(ch_dir + "/avg_r.npy", pattern.avg_r.data(), width, height);
    write_npy_uint16(ch_dir + "/blend0_avg.npy", pattern.blend0_avg.data(), width, height);
    write_npy_uint16(ch_dir + "/blend1_avg.npy", pattern.blend1_avg.data(), width, height);

    // Stage 4
    write_npy_uint16(ch_dir + "/blend0_iir.npy", pattern.blend0_iir.data(), width, height);
    write_npy_uint16(ch_dir + "/blend1_iir.npy", pattern.blend1_iir.data(), width, height);
    write_npy_uint16(ch_dir + "/final_output.npy", pattern.final_output.data(), width, height);

    return true;
}

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "CSIIR Pattern Output Testbench (Full Pipeline)\n"
                  << "Usage: " << argv[0] << " <input.bin> <output_dir> <width> <height>\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_dir = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    std::cout << "============================================================\n"
              << "CSIIR Pattern Output Testbench (Full Pipeline)\n"
              << "============================================================\n"
              << "Input:      " << input_file << "\n"
              << "Output dir: " << output_dir << "\n"
              << "Size:       " << width << " x " << height << "\n"
              << "Pixel:      " << PIXEL_BITWIDTH << "-bit\n"
              << "============================================================\n";

    // Create output directories
    create_directory(output_dir);
    create_directory(std::string(output_dir) + "/Y");
    create_directory(std::string(output_dir) + "/U");
    create_directory(std::string(output_dir) + "/V");

    // Read input
    BinaryHeader header;
    std::vector<uint16_t> y_in, u_in, v_in;

    std::cout << "\nReading input file...\n";
    if (!read_binary_input(input_file, y_in, u_in, v_in, header)) {
        std::cerr << "Error reading input\n";
        return 1;
    }

    uint16_t pixel_max = (1 << header.pixel_bits) - 1;

    // Process each channel
    std::cout << "\nProcessing channels with full pipeline...\n";

    ChannelPatternData y_pat, u_pat, v_pat;
    std::vector<uint16_t> y_out, u_out, v_out;

    process_channel_full_pipeline(y_in, y_out, y_pat, width, height, pixel_max);
    process_channel_full_pipeline(u_in, u_out, u_pat, width, height, pixel_max);
    process_channel_full_pipeline(v_in, v_out, v_pat, width, height, pixel_max);

    // Save pattern data
    std::cout << "\nSaving pattern data...\n";
    save_pattern_data(output_dir, "Y", y_pat, width, height);
    save_pattern_data(output_dir, "U", u_pat, width, height);
    save_pattern_data(output_dir, "V", v_pat, width, height);

    // Save config
    std::ofstream cfg(std::string(output_dir) + "/config.json");
    cfg << "{\n"
        << "  \"width\": " << width << ",\n"
        << "  \"height\": " << height << ",\n"
        << "  \"pixel_bits\": " << (int)header.pixel_bits << ",\n"
        << "  \"channels\": 3,\n"
        << "  \"model\": \"cpp_full_pipeline\"\n"
        << "}\n";

    std::cout << "\n============================================================\n"
              << "PATTERN OUTPUT COMPLETED SUCCESSFULLY\n"
              << "============================================================\n";

    return 0;
}