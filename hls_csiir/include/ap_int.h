/**
 * @file ap_int.h
 * @brief HLS 任意精度整数类型模拟 (简化版，用于独立编译测试)
 *
 * 注意: 仅用于 C-Simulation 验证，不用于实际综合
 */

#ifndef AP_INT_H
#define AP_INT_H

#include <cstdint>
#include <type_traits>

// 简化实现: 直接使用 typedef
typedef int8_t    ap_int8;
typedef int16_t   ap_int16;
typedef int32_t   ap_int32;
typedef int64_t   ap_int64;

typedef uint8_t   ap_uint8;
typedef uint16_t  ap_uint16;
typedef uint32_t  ap_uint32;
typedef uint64_t  ap_uint64;

// 模板别名 (简化版)
template <int W>
using ap_int = typename std::conditional<W <= 8, int8_t,
    typename std::conditional<W <= 16, int16_t,
        typename std::conditional<W <= 32, int32_t, int64_t>::type>::type>::type;

template <int W>
using ap_uint = typename std::conditional<W <= 8, uint8_t,
    typename std::conditional<W <= 16, uint16_t,
        typename std::conditional<W <= 32, uint32_t, uint64_t>::type>::type>::type;

#endif // AP_INT_H