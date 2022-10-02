#pragma once

#include "dim.h"

using namespace vision;

template<typename T>
bool dim_equal(const Dim<T> &lhs, const Dim<T> &rhs) {
    if (lhs._dims != rhs._dims or lhs._size != rhs._size) {
        return false;
    }
    if (lhs._ptr == rhs._ptr) {
        return true;
    }
    for (auto i = 0; i < lhs._size; ++i) {
        if (lhs._ptr[i] != rhs._ptr[i]) {
            return false;
        }
    }
    return true;
}