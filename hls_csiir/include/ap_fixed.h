/**
 * @file ap_fixed.h
 * @brief HLS 定点数类型模拟 (用于无 Vivado 环境编译测试)
 */

#ifndef AP_FIXED_H
#define AP_FIXED_H

// 简化的定点数类型 (暂不使用)
template <int W, int I>
class ap_fixed {
private:
    double value;

public:
    ap_fixed() : value(0) {}
    ap_fixed(double v) : value(v) {}

    operator double() const { return value; }
};

#endif // AP_FIXED_H