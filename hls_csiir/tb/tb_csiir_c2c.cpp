/**
 * @file tb_csiir_c2c.cpp
 * @brief CSIIR C2C (C-to-C) Validation Testbench
 *
 * Binary I/O testbench for Python <-> C++ data exchange.
 * Reads binary input file, processes through CSIIR, writes binary output file.
 *
 * Binary Format (matching Python csiir_c2c_utils.py):
 *   Header (16 bytes):
 *     - magic: "CSII" (4 bytes)
 *     - width: uint16
 *     - height: uint16
 *     - pixel_bits: uint8
 *     - channels: uint8
 *     - reserved: 6 bytes
 *   Data:
 *     - Raw pixels in row-major order: Y0,U0,V0, Y1,U1,V1, ...
 *
 * Usage: tb_csiir_c2c <input.bin> <output.bin> <width> <height> [debug_dir]
 *
 * @version 1.0
 * @date 2026-03-15
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
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

    // Read header
    file.read(reinterpret_cast<char*>(&header), HEADER_SIZE);
    if (file.gcount() != HEADER_SIZE) {
        std::cerr << "Error: Failed to read header" << std::endl;
        return false;
    }

    // Validate magic
    if (strncmp(header.magic, MAGIC, 4) != 0) {
        std::cerr << "Error: Invalid magic number. Expected 'CSII', got '"
                  << std::string(header.magic, 4) << "'" << std::endl;
        return false;
    }

    // Validate pixel bits
    if (header.pixel_bits != 8 && header.pixel_bits != 10 && header.pixel_bits != 12) {
        std::cerr << "Error: Unsupported pixel_bits: " << (int)header.pixel_bits << std::endl;
        return false;
    }

    // Validate channels
    if (header.channels != 3) {
        std::cerr << "Error: Only 3-channel (YUV) supported. Got: " << (int)header.channels << std::endl;
        return false;
    }

    size_t total_pixels = (size_t)header.width * header.height;
    y_data.resize(total_pixels);
    u_data.resize(total_pixels);
    v_data.resize(total_pixels);

    // Read pixel data
    if (header.pixel_bits <= 8) {
        // 8-bit pixels
        std::vector<uint8_t> raw(total_pixels * 3);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size());
        if (file.gcount() != (std::streamsize)raw.size()) {
            std::cerr << "Error: Failed to read pixel data" << std::endl;
            return false;
        }
        for (size_t i = 0; i < total_pixels; i++) {
            y_data[i] = raw[i * 3 + 0];
            u_data[i] = raw[i * 3 + 1];
            v_data[i] = raw[i * 3 + 2];
        }
    } else {
        // 10/12-bit pixels (16-bit storage)
        std::vector<uint16_t> raw(total_pixels * 3);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size() * 2);
        if (file.gcount() != (std::streamsize)(raw.size() * 2)) {
            std::cerr << "Error: Failed to read pixel data" << std::endl;
            return false;
        }
        for (size_t i = 0; i < total_pixels; i++) {
            y_data[i] = raw[i * 3 + 0];
            u_data[i] = raw[i * 3 + 1];
            v_data[i] = raw[i * 3 + 2];
        }
    }

    return true;
}

bool write_binary_output(const char* filepath,
                         const std::vector<uint16_t>& y_data,
                         const std::vector<uint16_t>& u_data,
                         const std::vector<uint16_t>& v_data,
                         const BinaryHeader& header) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << filepath << std::endl;
        return false;
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&header), HEADER_SIZE);

    // Write pixel data
    size_t total_pixels = (size_t)header.width * header.height;
    if (header.pixel_bits <= 8) {
        // 8-bit pixels
        std::vector<uint8_t> raw(total_pixels * 3);
        for (size_t i = 0; i < total_pixels; i++) {
            raw[i * 3 + 0] = (uint8_t)std::min(y_data[i], (uint16_t)255);
            raw[i * 3 + 1] = (uint8_t)std::min(u_data[i], (uint16_t)255);
            raw[i * 3 + 2] = (uint8_t)std::min(v_data[i], (uint16_t)255);
        }
        file.write(reinterpret_cast<char*>(raw.data()), raw.size());
    } else {
        // 10/12-bit pixels
        std::vector<uint16_t> raw(total_pixels * 3);
        for (size_t i = 0; i < total_pixels; i++) {
            raw[i * 3 + 0] = y_data[i];
            raw[i * 3 + 1] = u_data[i];
            raw[i * 3 + 2] = v_data[i];
        }
        file.write(reinterpret_cast<char*>(raw.data()), raw.size() * 2);
    }

    return true;
}

//=============================================================================
// CSIIR Processing Wrapper
//=============================================================================

void process_csiir(const std::vector<uint16_t>& y_in,
                   const std::vector<uint16_t>& u_in,
                   const std::vector<uint16_t>& v_in,
                   std::vector<uint16_t>& y_out,
                   std::vector<uint16_t>& u_out,
                   std::vector<uint16_t>& v_out,
                   uint16_t width,
                   uint16_t height) {
    // Create AXI streams
    hls::stream<AxisYUV> axis_in("axis_in");
    hls::stream<AxisYUV> axis_out("axis_out");

    // Configuration (thresholds scaled for pixel bit depth)
    CSIIRConfig config;
    // Scale thresholds: base 8-bit threshold * 4 for 10-bit
    config.sobel_thresh_0 = 64;   // 2x2 window threshold
    config.sobel_thresh_1 = 96;   // 3x3 window threshold
    config.sobel_thresh_2 = 128;  // 4x4 window threshold
    config.sobel_thresh_3 = 160;  // 5x5 window threshold

    config.blend_coeff[0] = 32;
    config.blend_coeff[1] = 32;
    config.blend_coeff[2] = 32;
    config.blend_coeff[3] = 32;

    // Initialize output vectors
    size_t total_pixels = (size_t)width * height;
    y_out.resize(total_pixels);
    u_out.resize(total_pixels);
    v_out.resize(total_pixels);

    // Write input data to stream
    for (size_t i = 0; i < total_pixels; i++) {
        AxisYUV data;
        data.y = (pixel_t)y_in[i];
        data.u = (pixel_t)u_in[i];
        data.v = (pixel_t)v_in[i];
        data.user = (i == 0) ? 1 : 0;
        data.last = ((i % width) == width - 1) ? 1 : 0;
        axis_in.write(data);
    }

    // Process through CSIIR
    csiir_top(axis_in, axis_out, config, (index_t)width, (index_t)height);

    // Read output data from stream
    for (size_t i = 0; i < total_pixels; i++) {
        AxisYUV data = axis_out.read();
        y_out[i] = (uint16_t)data.y;
        u_out[i] = (uint16_t)data.u;
        v_out[i] = (uint16_t)data.v;
    }
}

//=============================================================================
// Main Entry Point
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "CSIIR C2C Validation Testbench\n"
              << "Usage: " << prog << " <input.bin> <output.bin> <width> <height> [debug_dir]\n\n"
              << "Arguments:\n"
              << "  input.bin   Input binary file (YUV444 format)\n"
              << "  output.bin  Output binary file\n"
              << "  width       Image width\n"
              << "  height      Image height\n"
              << "  debug_dir   Optional debug output directory\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    std::cout << "============================================================" << std::endl;
    std::cout << "CSIIR C2C Testbench" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Input:  " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    std::cout << "Size:   " << width << " x " << height << std::endl;
    std::cout << "Pixel:  " << PIXEL_BITWIDTH << "-bit" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Read input
    BinaryHeader header;
    std::vector<uint16_t> y_in, u_in, v_in;

    std::cout << "\nReading input file..." << std::endl;
    if (!read_binary_input(input_file, y_in, u_in, v_in, header)) {
        return 1;
    }

    // Verify dimensions
    if (header.width != width || header.height != height) {
        std::cout << "Warning: Header dimensions (" << header.width << "x" << header.height
                  << ") differ from command line (" << width << "x" << height << ")" << std::endl;
        width = header.width;
        height = header.height;
    }

    std::cout << "Read " << y_in.size() << " pixels per channel" << std::endl;

    // Process through CSIIR
    std::vector<uint16_t> y_out, u_out, v_out;

    std::cout << "\nProcessing through CSIIR..." << std::endl;
    process_csiir(y_in, u_in, v_in, y_out, u_out, v_out, width, height);
    std::cout << "Processed " << y_out.size() << " pixels per channel" << std::endl;

    // Write output
    std::cout << "\nWriting output file..." << std::endl;
    if (!write_binary_output(output_file, y_out, u_out, v_out, header)) {
        return 1;
    }
    std::cout << "Output written to: " << output_file << std::endl;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "C2C TESTBENCH COMPLETED SUCCESSFULLY" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}