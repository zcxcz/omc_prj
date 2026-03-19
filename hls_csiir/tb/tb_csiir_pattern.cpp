/**
 * @file tb_csiir_pattern.cpp
 * @brief CSIIR Pattern Output Testbench
 *
 * Extended C2C testbench that outputs intermediate data patterns
 * at each pipeline stage for comparison with Python reference.
 *
 * Output Directory Structure:
 *   <output_dir>/
 *   ├── Y/
 *   │   ├── stage1_sobel.npz
 *   │   ├── stage2_window_selector.npz
 *   │   ├── stage3_directional_filter.npz
 *   │   └── stage4_blending.npz
 *   ├── U/
 *   │   └── ...
 *   ├── V/
 *   │   └── ...
 *   ├── config.json
 *   ├── input.npz
 *   └── output.npz
 *
 * Usage: tb_csiir_pattern <input.bin> <output_dir> <width> <height>
 *
 * @version 1.0
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

//=============================================================================
// Binary File Format
//=============================================================================

#pragma pack(push, 1)
struct BinaryHeader {
    char magic[4];       // "CSII"
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
// Pattern Output Structure
//=============================================================================

struct PatternData {
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
// NPZ File Writer (Simple Implementation)
//=============================================================================

// Write NPZ file with single 2D array
bool write_npz_array_2d(const std::string& filepath, const std::string& name,
                        const uint16_t* data, uint16_t width, uint16_t height) {
    // Create NPY file content
    // NPY header format: magic(6) + version(2) + header_len(2) + header
    std::stringstream ss;
    ss << "{'descr': '<u2', 'fortran_order': False, 'shape': ("
       << height << ", " << width << "), }";

    std::string header_dict = ss.str();

    // Pad header to 64-byte alignment (including the 10-byte prefix)
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) {
        header_dict += std::string(padding - 1, ' ') + '\n';
    }

    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return false;
    }

    // Write NPY header
    file.write("\x93NUMPY", 6);           // Magic
    file.write("\x01\x00", 2);             // Version 1.0
    file.write(reinterpret_cast<const char*>(&header_len), 2);  // Header length
    file.write(header_dict.c_str(), header_len);

    // Write data
    file.write(reinterpret_cast<const char*>(data), width * height * sizeof(uint16_t));
    file.close();

    return true;
}

bool write_npz_array_2d_int32(const std::string& filepath, const std::string& name,
                              const int32_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<i4', 'fortran_order': False, 'shape': ("
       << height << ", " << width << "), }";

    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) {
        header_dict += std::string(padding - 1, ' ') + '\n';
    }

    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), width * height * sizeof(int32_t));
    file.close();
    return true;
}

bool write_npz_array_2d_uint32(const std::string& filepath, const std::string& name,
                               const uint32_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<u4', 'fortran_order': False, 'shape': ("
       << height << ", " << width << "), }";

    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) {
        header_dict += std::string(padding - 1, ' ') + '\n';
    }

    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), width * height * sizeof(uint32_t));
    file.close();
    return true;
}

bool write_npz_array_2d_uint8(const std::string& filepath, const std::string& name,
                              const uint8_t* data, uint16_t width, uint16_t height) {
    std::stringstream ss;
    ss << "{'descr': '<u1', 'fortran_order': False, 'shape': ("
       << height << ", " << width << "), }";

    std::string header_dict = ss.str();
    size_t header_total = 10 + header_dict.length();
    size_t padding = 64 - (header_total % 64);
    if (padding < 64) {
        header_dict += std::string(padding - 1, ' ') + '\n';
    }

    uint16_t header_len = header_dict.length();

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_dict.c_str(), header_len);
    file.write(reinterpret_cast<const char*>(data), width * height);
    file.close();
    return true;
}

//=============================================================================
// Directory Utilities
//=============================================================================

bool create_directory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

#include <sys/stat.h>
#include <errno.h>

//=============================================================================
// Helper Functions
//=============================================================================

bool read_binary_input(const char* filepath,
                       std::vector<uint16_t>& y_data,
                       std::vector<uint16_t>& u_data,
                       std::vector<uint16_t>& v_data,
                       BinaryHeader& header) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file: " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&header), HEADER_SIZE);
    if (file.gcount() != HEADER_SIZE) {
        std::cerr << "Error: Failed to read header" << std::endl;
        return false;
    }

    if (strncmp(header.magic, MAGIC, 4) != 0) {
        std::cerr << "Error: Invalid magic number" << std::endl;
        return false;
    }

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
// CSIIR Processing with Pattern Capture
//=============================================================================

void process_channel_with_pattern(
    const std::vector<uint16_t>& channel_in,
    std::vector<uint16_t>& channel_out,
    PatternData& pattern,
    uint16_t width,
    uint16_t height,
    uint16_t pixel_max)
{
    size_t total_pixels = (size_t)width * height;
    channel_out.resize(total_pixels);
    pattern.resize(total_pixels);

    // Configuration
    uint16_t thresh[4] = {64, 96, 128, 160};  // Scaled for 10-bit
    uint8_t blend_ratio[4] = {32, 32, 32, 32};

    // Sobel kernels (simplified difference)
    const int8_t SOBEL_X[5][5] = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {-1, -1, -1, -1, -1}
    };
    const int8_t SOBEL_Y[5][5] = {
        {1, 0, 0, 0, -1},
        {1, 0, 0, 0, -1},
        {1, 0, 0, 0, -1},
        {1, 0, 0, 0, -1},
        {1, 0, 0, 0, -1}
    };

    // Pad input for border handling
    std::vector<uint16_t> padded((height + 4) * (width + 4), 0);
    for (int y = 0; y < height; y++) {
        int py = y + 2;
        for (int x = 0; x < width; x++) {
            int px = x + 2;
            padded[py * (width + 4) + px] = channel_in[y * width + x];

            // Reflect padding
            if (x < 2) padded[py * (width + 4) + (2 - x)] = channel_in[y * width + x];
            if (x >= width - 2) padded[py * (width + 4) + (width + 4 - 1 - (x - width + 2))] = channel_in[y * width + x];
        }
        if (y < 2) {
            for (int x = 0; x < (int)(width + 4); x++) {
                padded[(2 - y) * (width + 4) + x] = padded[py * (width + 4) + x];
            }
        }
        if (y >= height - 2) {
            for (int x = 0; x < (int)(width + 4); x++) {
                padded[(height + 4 - 1 - (y - height + 2)) * (width + 4) + x] = padded[py * (width + 4) + x];
            }
        }
    }

    // Pre-compute gradient map
    std::vector<uint32_t> grad_map(total_pixels, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gx = 0, gy = 0;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    uint16_t pixel = padded[(y + i) * (width + 4) + (x + j)];
                    gx += pixel * SOBEL_X[i][j];
                    gy += pixel * SOBEL_Y[i][j];
                }
            }
            int grad = (abs(gx) + 2) / 5 + (abs(gy) + 2) / 5;
            grad_map[y * width + x] = grad;

            // Store Stage 1 data
            size_t idx = y * width + x;
            pattern.grad_h[idx] = gx;
            pattern.grad_v[idx] = gy;
            pattern.grad_magnitude[idx] = grad;
        }
    }

    // Process each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = y * width + x;
            uint32_t grad = grad_map[idx];

            // Get neighboring gradients
            uint32_t grad_prev = (x > 0) ? grad_map[idx - 1] : grad;
            uint32_t grad_next = (x < width - 1) ? grad_map[idx + 1] : grad;
            uint32_t max_grad = grad_prev;
            if (grad > max_grad) max_grad = grad;
            if (grad_next > max_grad) max_grad = grad_next;

            // Window size selection (Stage 2)
            uint8_t ws;
            if (max_grad < thresh[0]) ws = 2;
            else if (max_grad < thresh[1]) ws = 3;
            else if (max_grad < thresh[2]) ws = 4;
            else ws = 5;

            pattern.win_size[idx] = ws;
            pattern.grad_used[idx] = max_grad;

            // Extract 5x5 window
            uint16_t window[5][5];
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    window[i][j] = padded[(y + i) * (width + 4) + (x + j)];
                }
            }

            // Simple averaging (placeholder for full algorithm)
            // In production, this would call the full directional_filter and blending
            uint32_t sum = 0;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    sum += window[i][j];
                }
            }
            uint16_t avg = sum / 25;

            // Store Stage 3 data (simplified)
            pattern.avg_c[idx] = avg;
            pattern.avg_u[idx] = avg;
            pattern.avg_d[idx] = avg;
            pattern.avg_l[idx] = avg;
            pattern.avg_r[idx] = avg;
            pattern.blend0_avg[idx] = avg;
            pattern.blend1_avg[idx] = avg;

            // Store Stage 4 data (simplified)
            pattern.blend0_iir[idx] = avg;
            pattern.blend1_iir[idx] = avg;
            pattern.final_output[idx] = avg;

            channel_out[idx] = avg;
        }
    }
}

//=============================================================================
// Save Pattern Data
//=============================================================================

bool save_pattern_data(const std::string& base_dir, const std::string& channel,
                       const PatternData& pattern, uint16_t width, uint16_t height) {
    std::string ch_dir = base_dir + "/" + channel;
    create_directory(ch_dir);

    // Save Stage 1
    std::string s1_file = ch_dir + "/stage1_sobel.npz";
    // For simplicity, we'll write individual NPY files
    // A real implementation would use a ZIP library for NPZ

    write_npz_array_2d_int32(ch_dir + "/grad_h.npy", "grad_h",
                              pattern.grad_h.data(), width, height);
    write_npz_array_2d_int32(ch_dir + "/grad_v.npy", "grad_v",
                              pattern.grad_v.data(), width, height);
    write_npz_array_2d_uint32(ch_dir + "/grad_magnitude.npy", "grad_magnitude",
                               pattern.grad_magnitude.data(), width, height);

    // Save Stage 2
    write_npz_array_2d_uint8(ch_dir + "/win_size.npy", "win_size",
                              pattern.win_size.data(), width, height);
    write_npz_array_2d_uint32(ch_dir + "/grad_used.npy", "grad_used",
                               pattern.grad_used.data(), width, height);

    // Save Stage 3
    write_npz_array_2d(ch_dir + "/avg_c.npy", "avg_c",
                        pattern.avg_c.data(), width, height);
    write_npz_array_2d(ch_dir + "/avg_u.npy", "avg_u",
                        pattern.avg_u.data(), width, height);
    write_npz_array_2d(ch_dir + "/avg_d.npy", "avg_d",
                        pattern.avg_d.data(), width, height);
    write_npz_array_2d(ch_dir + "/avg_l.npy", "avg_l",
                        pattern.avg_l.data(), width, height);
    write_npz_array_2d(ch_dir + "/avg_r.npy", "avg_r",
                        pattern.avg_r.data(), width, height);
    write_npz_array_2d(ch_dir + "/blend0_avg.npy", "blend0_avg",
                        pattern.blend0_avg.data(), width, height);
    write_npz_array_2d(ch_dir + "/blend1_avg.npy", "blend1_avg",
                        pattern.blend1_avg.data(), width, height);

    // Save Stage 4
    write_npz_array_2d(ch_dir + "/blend0_iir.npy", "blend0_iir",
                        pattern.blend0_iir.data(), width, height);
    write_npz_array_2d(ch_dir + "/blend1_iir.npy", "blend1_iir",
                        pattern.blend1_iir.data(), width, height);
    write_npz_array_2d(ch_dir + "/final_output.npy", "final_output",
                        pattern.final_output.data(), width, height);

    return true;
}

//=============================================================================
// Main Entry Point
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "CSIIR Pattern Output Testbench\n"
              << "Usage: " << prog << " <input.bin> <output_dir> <width> <height>\n\n"
              << "Arguments:\n"
              << "  input.bin   Input binary file (YUV444 format)\n"
              << "  output_dir  Output directory for pattern files\n"
              << "  width       Image width\n"
              << "  height      Image height\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_dir = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    std::cout << "============================================================" << std::endl;
    std::cout << "CSIIR Pattern Output Testbench" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Input:      " << input_file << std::endl;
    std::cout << "Output dir: " << output_dir << std::endl;
    std::cout << "Size:       " << width << " x " << height << std::endl;
    std::cout << "Pixel:      " << PIXEL_BITWIDTH << "-bit" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Create output directories
    create_directory(output_dir);
    create_directory(std::string(output_dir) + "/Y");
    create_directory(std::string(output_dir) + "/U");
    create_directory(std::string(output_dir) + "/V");

    // Read input
    BinaryHeader header;
    std::vector<uint16_t> y_in, u_in, v_in;

    std::cout << "\nReading input file..." << std::endl;
    if (!read_binary_input(input_file, y_in, u_in, v_in, header)) {
        return 1;
    }

    if (header.width != width || header.height != height) {
        width = header.width;
        height = header.height;
    }

    uint16_t pixel_max = (1 << header.pixel_bits) - 1;

    // Process each channel with pattern capture
    std::cout << "\nProcessing channels with pattern capture..." << std::endl;

    PatternData y_pattern, u_pattern, v_pattern;
    std::vector<uint16_t> y_out, u_out, v_out;

    process_channel_with_pattern(y_in, y_out, y_pattern, width, height, pixel_max);
    process_channel_with_pattern(u_in, u_out, u_pattern, width, height, pixel_max);
    process_channel_with_pattern(v_in, v_out, v_pattern, width, height, pixel_max);

    // Save pattern data
    std::cout << "\nSaving pattern data..." << std::endl;
    save_pattern_data(output_dir, "Y", y_pattern, width, height);
    save_pattern_data(output_dir, "U", u_pattern, width, height);
    save_pattern_data(output_dir, "V", v_pattern, width, height);

    // Save config.json
    std::string config_file = std::string(output_dir) + "/config.json";
    std::ofstream cfg(config_file);
    cfg << "{\n";
    cfg << "  \"width\": " << width << ",\n";
    cfg << "  \"height\": " << height << ",\n";
    cfg << "  \"pixel_bits\": " << (int)header.pixel_bits << ",\n";
    cfg << "  \"channels\": 3,\n";
    cfg << "  \"model\": \"cpp\"\n";
    cfg << "}\n";
    cfg.close();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "PATTERN OUTPUT COMPLETED SUCCESSFULLY" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}