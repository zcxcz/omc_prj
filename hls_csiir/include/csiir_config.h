/**
 * @file csiir_config.h
 * @brief CSIIR 模块参数化配置
 *
 * HLS CSIIR 模块 - 编译时可配置参数
 *
 * @version 2.0
 * @date 2026-03-13
 */

#ifndef CSIIR_CONFIG_H
#define CSIIR_CONFIG_H

//=============================================================================
// 像素位宽配置 (默认 10-bit)
//=============================================================================

/**
 * @brief 像素位宽
 * 支持 8-bit, 10-bit, 12-bit 等
 * 可通过编译选项覆盖: -DPIXEL_BITWIDTH=10
 */
#ifndef PIXEL_BITWIDTH
#define PIXEL_BITWIDTH 10
#endif

/**
 * @brief 像素最大值
 */
#define PIXEL_MAX  ((1 << PIXEL_BITWIDTH) - 1)

//=============================================================================
// 最大分辨率配置 (默认 8K)
//=============================================================================

/**
 * @brief 最大图像宽度
 * 支持: 1920 (1080p), 3840 (4K), 7680 (8K)
 * 可通过编译选项覆盖: -DMAX_IMAGE_WIDTH=7680
 */
#ifndef MAX_IMAGE_WIDTH
#define MAX_IMAGE_WIDTH 7680
#endif

/**
 * @brief 最大图像高度
 * 支持: 1080 (1080p), 2160 (4K), 4320 (8K)
 * 可通过编译选项覆盖: -DMAX_IMAGE_HEIGHT=4320
 */
#ifndef MAX_IMAGE_HEIGHT
#define MAX_IMAGE_HEIGHT 4320
#endif

/**
 * @brief 索引位宽计算
 * 支持最大索引值: 8K = 7680, 需要 13-bit (最大 8191)
 * 使用 14-bit 以留有余量
 */
#define INDEX_BITWIDTH 14

//=============================================================================
// YUV 格式配置
//=============================================================================

/**
 * @brief YUV 格式选择
 * 支持: 444, 422, 420
 * 可通过编译选项覆盖: -DYUV_FORMAT=422
 */
#ifndef YUV_FORMAT
#define YUV_FORMAT 422  // 默认 YUV422
#endif

/**
 * @brief 根据 YUV 格式定义通道数和下采样参数
 */
#if YUV_FORMAT == 444
    /** YUV444: Y, U, V 全部独立，无下采样 */
    #define YUV_CHANNELS 3
    #define UV_SUBSAMPLE 1

#elif YUV_FORMAT == 422
    /** YUV422: UV 水平 2:1 下采样 */
    #define YUV_CHANNELS 2
    #define UV_SUBSAMPLE 2

#elif YUV_FORMAT == 420
    /** YUV420: UV 水平+垂直 2:1 下采样 */
    #define YUV_CHANNELS 2
    #define UV_SUBSAMPLE 4

#else
    #error "Unsupported YUV_FORMAT. Use 444, 422, or 420."
#endif

//=============================================================================
// 通道处理模式
//=============================================================================

/**
 * @brief CSIIR 处理模式
 * 1: 所有通道独立进行 CSIIR 处理 (Y/U/V 各通道独立)
 * 0: 仅 UV 双通道处理 (向后兼容模式)
 */
#define CSIIR_PROCESS_ALL_CHANNELS  1

//=============================================================================
// Line Buffer 配置
//=============================================================================

/**
 * @brief UV Line Buffer 行数
 */
#ifndef UV_LINE_BUFFER_ROWS
#define UV_LINE_BUFFER_ROWS      6
#endif

/**
 * @brief 梯度 Line Buffer 行数
 */
#ifndef GRAD_LINE_BUFFER_ROWS
#define GRAD_LINE_BUFFER_ROWS    5
#endif

//=============================================================================
// 窗口大小常量
//=============================================================================

#define WINSIZE_2x2              2
#define WINSIZE_3x3              3
#define WINSIZE_4x4              4
#define WINSIZE_5x5              5

//=============================================================================
// 编译时断言 (静态检查)
//=============================================================================

/**
 * @brief 检查索引位宽是否足够
 * MAX_IMAGE_WIDTH 需要小于 2^INDEX_BITWIDTH
 */
#if MAX_IMAGE_WIDTH >= (1 << INDEX_BITWIDTH)
    #error "INDEX_BITWIDTH too small for MAX_IMAGE_WIDTH"
#endif

#if MAX_IMAGE_HEIGHT >= (1 << INDEX_BITWIDTH)
    #error "INDEX_BITWIDTH too small for MAX_IMAGE_HEIGHT"
#endif

#endif // CSIIR_CONFIG_H