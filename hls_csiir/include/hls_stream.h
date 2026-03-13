/**
 * @file hls_stream.h
 * @brief HLS 流类型模拟 (用于无 Vivado 环境编译测试)
 */

#ifndef HLS_STREAM_H
#define HLS_STREAM_H

#include <queue>
#include <string>
#include <cstddef>

namespace hls {

template <typename T>
class stream {
private:
    std::queue<T> q;
    std::string name;

public:
    stream() {}
    stream(const char* n) : name(n) {}

    void write(const T& val) {
        q.push(val);
    }

    T read() {
        T val = q.front();
        q.pop();
        return val;
    }

    bool empty() const {
        return q.empty();
    }

    bool full() const {
        return false;  // 简化实现
    }

    size_t size() const {
        return q.size();
    }
};

} // namespace hls

#endif // HLS_STREAM_H