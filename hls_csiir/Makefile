# =============================================================================
# CSIIR HLS Module Makefile
# =============================================================================
#
# Build targets for CSIIR module:
#   - C++ simulation testbench
#   - C2C validation testbench
#   - HLS synthesis (Vivado HLS)
#   - Cleaning targets
#
# Usage:
#   make c2c          - Compile C2C testbench
#   make run-c2c      - Run C2C validation
#   make sim          - Run basic simulation
#   make clean        - Clean build artifacts
#
# =============================================================================

# Configuration
HLS_DIR := .
SRC_DIR := $(HLS_DIR)/src
TB_DIR := $(HLS_DIR)/tb
INC_DIR := $(HLS_DIR)/include

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++11 -O2 -Wall
LDFLAGS :=

# Default configuration
PIXEL_BITWIDTH ?= 10
YUV_FORMAT ?= 444
MAX_WIDTH ?= 7680
MAX_HEIGHT ?= 4320

# Derived flags
DEFINES := -DPIXEL_BITWIDTH=$(PIXEL_BITWIDTH) \
           -DYUV_FORMAT=$(YUV_FORMAT) \
           -DMAX_IMAGE_WIDTH=$(MAX_WIDTH) \
           -DMAX_IMAGE_HEIGHT=$(MAX_HEIGHT)

INCLUDES := -I$(INC_DIR)

# Source files
SRCS := $(SRC_DIR)/sobel_filter.cpp \
        $(SRC_DIR)/window_selector.cpp \
        $(SRC_DIR)/directional_filter.cpp \
        $(SRC_DIR)/blending.cpp \
        $(SRC_DIR)/csiir_top.cpp

# =============================================================================
# Targets
# =============================================================================

.PHONY: all c2c run-c2c sim clean help

all: c2c

# -----------------------------------------------------------------------------
# C2C Validation Testbench
# -----------------------------------------------------------------------------

# C2C binary names based on configuration
C2C_BIN := tb_csiir_c2c
C2C_BIN_8BIT := tb_csiir_c2c_8bit
C2C_BIN_10BIT := tb_csiir_c2c_10bit

# Compile C2C testbench (default 10-bit)
c2c: $(C2C_BIN)_$(PIXEL_BITWIDTH)bit

# Compile 8-bit C2C testbench
c2c-8bit:
	@echo "Compiling C2C testbench (8-bit)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPIXEL_BITWIDTH=8 $(DEFINES) \
		$(SRCS) $(TB_DIR)/tb_csiir_c2c.cpp \
		-o $(C2C_BIN)_8bit
	@echo "Built: $(C2C_BIN)_8bit"

# Compile 10-bit C2C testbench
c2c-10bit:
	@echo "Compiling C2C testbench (10-bit)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPIXEL_BITWIDTH=10 $(DEFINES) \
		$(SRCS) $(TB_DIR)/tb_csiir_c2c.cpp \
		-o $(C2C_BIN)_10bit
	@echo "Built: $(C2C_BIN)_10bit"

# Compile 12-bit C2C testbench
c2c-12bit:
	@echo "Compiling C2C testbench (12-bit)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPIXEL_BITWIDTH=12 $(DEFINES) \
		$(SRCS) $(TB_DIR)/tb_csiir_c2c.cpp \
		-o $(C2C_BIN)_12bit
	@echo "Built: $(C2C_BIN)_12bit"

# Compile all bit-width variants
c2c-all: c2c-8bit c2c-10bit c2c-12bit

# -----------------------------------------------------------------------------
# Run C2C Validation
# -----------------------------------------------------------------------------

run-c2c:
	@echo "Running C2C validation..."
	cd .. && python3 algorithms/run_c2c_validation.py --patterns all --pixel-bits $(PIXEL_BITWIDTH)

run-c2c-8bit:
	@echo "Running C2C validation (8-bit)..."
	cd .. && python3 algorithms/run_c2c_validation.py --patterns all --pixel-bits 8

run-c2c-10bit:
	@echo "Running C2C validation (10-bit)..."
	cd .. && python3 algorithms/run_c2c_validation.py --patterns all --pixel-bits 10

run-c2c-12bit:
	@echo "Running C2C validation (12-bit)..."
	cd .. && python3 algorithms/run_c2c_validation.py --patterns all --pixel-bits 12

# Run C2C with custom patterns
run-c2c-custom:
	@if [ -z "$(PATTERNS)" ]; then \
		echo "Usage: make run-c2c-custom PATTERNS=flat,gradient,edge"; \
		exit 1; \
	fi
	cd .. && python3 algorithms/run_c2c_validation.py --patterns $(PATTERNS) --pixel-bits $(PIXEL_BITWIDTH)

# -----------------------------------------------------------------------------
# Basic Simulation Testbench
# -----------------------------------------------------------------------------

SIM_BIN := tb_csiir

sim: $(SIM_BIN)
	@echo "Running basic simulation..."
	./$(SIM_BIN)

$(SIM_BIN): $(SRCS) $(TB_DIR)/tb_csiir.cpp
	@echo "Compiling simulation testbench..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) \
		$(SRCS) $(TB_DIR)/tb_csiir.cpp \
		-o $(SIM_BIN)
	@echo "Built: $(SIM_BIN)"

# Simple testbench (minimal)
sim-simple:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) \
		$(SRCS) $(TB_DIR)/tb_csiir_simple.cpp \
		-o tb_csiir_simple
	./tb_csiir_simple

# -----------------------------------------------------------------------------
# HLS Synthesis (Vivado HLS)
# -----------------------------------------------------------------------------

HLS ?= vivado_hls
HLS_TOP := csiir_top
HLS_SOLUTION := solution1
HLS_DEVICE := xc7z020clg400-1
HLS_PERIOD := 10

# HLS synthesis
hls-synth:
	@echo "Running HLS synthesis..."
	$(HLS) -f scripts/run_hls.tcl

# HLS cosimulation
hls-cosim:
	@echo "Running HLS cosimulation..."
	$(HLS) -f scripts/run_hls_cosim.tcl

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(C2C_BIN) $(C2C_BIN)_8bit $(C2C_BIN)_10bit $(C2C_BIN)_12bit
	rm -f $(SIM_BIN) tb_csiir_simple tb_csiir_444 tb_csiir_8bit
	rm -rf c2c_data/*.bin c2c_data/*.npz c2c_data/*.txt
	rm -rf .hls
	@echo "Cleaned."

clean-data:
	@echo "Cleaning C2C data files..."
	rm -rf c2c_data/*
	@echo "Cleaned C2C data."

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo "CSIIR HLS Module Makefile"
	@echo ""
	@echo "Building:"
	@echo "  make c2c          - Compile C2C testbench (default 10-bit)"
	@echo "  make c2c-8bit     - Compile 8-bit C2C testbench"
	@echo "  make c2c-10bit    - Compile 10-bit C2C testbench"
	@echo "  make c2c-12bit    - Compile 12-bit C2C testbench"
	@echo "  make c2c-all      - Compile all bit-width variants"
	@echo "  make sim          - Compile and run basic simulation"
	@echo ""
	@echo "Running C2C Validation:"
	@echo "  make run-c2c      - Run C2C validation (default 10-bit)"
	@echo "  make run-c2c-8bit - Run 8-bit C2C validation"
	@echo "  make run-c2c-10bit- Run 10-bit C2C validation"
	@echo "  make run-c2c-12bit- Run 12-bit C2C validation"
	@echo "  make run-c2c-custom PATTERNS=flat,gradient - Custom patterns"
	@echo ""
	@echo "HLS Synthesis:"
	@echo "  make hls-synth    - Run Vivado HLS synthesis"
	@echo "  make hls-cosim    - Run HLS cosimulation"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make clean-data   - Remove C2C data files only"
	@echo ""
	@echo "Configuration:"
	@echo "  PIXEL_BITWIDTH=$(PIXEL_BITWIDTH)"
	@echo "  YUV_FORMAT=$(YUV_FORMAT)"
	@echo "  MAX_WIDTH=$(MAX_WIDTH)"
	@echo "  MAX_HEIGHT=$(MAX_HEIGHT)"

# =============================================================================
# End of Makefile
# =============================================================================