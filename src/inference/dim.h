#ifndef _DIM_H_
#define _DIM_H_

#include <algorithm>
#include <cstdint> // for uint32_t
#include <initializer_list>
#include <vector>

namespace vision {

/**
 * @brief Dim contains dimension information and (optional) raw data of a
 * tensor.
 *
 * @tparam T the type of raw data. Must be float by now.
 */
template <typename T> struct Dim {
  using value_type = T;
  std::vector<int> _dims;
  int _size = 0;
  T *_ptr = nullptr;

  /**
   * @brief Construct a zero-sized, no-memory-allocated new Dim object.
   */
  Dim() : _dims(), _size(0), _ptr(nullptr) {}

  /**
   * @brief Construct a new Dim object from initializer list.
   *
   * @param dims dimension size per channel.
   *
   * @example auto dim = Dim<float>{1, 3, 64, 64};
   */
  Dim(std::initializer_list<int> dims) : _dims(dims) {
    // dims was set in member initializer list
    // just update _size and allocate memeory
    this->_allocate();
  }

  /**
   * @brief Construct a new Dim object by given dimensions 
   * 
   * @param len length of \c pi
   * @param pi address of \c len continuous elements
   */
  Dim(int len, const int *pi) : _dims(len) {
    // set _dims by coping
    std::copy_n(pi, len, _dims.begin());
    // update _size and allocate memeory
    this->_allocate();
  }

  Dim(const Dim<T> &other) : _dims(other._dims), _size(other._size) {
    // _dims & _size was set in member initializer list
    // just update _size and allocate memeory
    this->_allocate();
    // then copy data from other (if any)
    if (other._ptr != nullptr) {
      std::copy_n(other._ptr, other._size, _ptr);
    }
  }

  Dim(Dim<T>&& rvalue) noexcept {
    // move assets
    _dims = std::forward<std::vector<int>>(rvalue._dims);
    _size = rvalue._size;
    _ptr = rvalue._ptr;
    // prevent _ptr get freed when rvalue is deconstructed
    rvalue._size = 0;
    rvalue._ptr = nullptr;
  }

  Dim<T> &operator=(const Dim<T> &other) {
    // copy _dim
    _dims.resize(other._dims.size());
    std::copy(other._dims.cbegin(), other._dims.cend(), _dims.begin());
    // copy _size
    this->_size = other._size;
    // copy _ptr
    if (other._ptr != nullptr) {
      this->_allocate();
      std::copy_n(other._ptr, other._size, _ptr);
    }
    return *this;
  }

  Dim<T> &operator=(Dim<T>&& rvalue) noexcept {
    // move assets
    _dims = std::forward<std::vector<int>>(rvalue._dims);
    _size = rvalue._size;
    _ptr = rvalue._ptr;
    // prevent _ptr get freed when rvalue is deconstructed
    rvalue._size = 0;
    rvalue._ptr = nullptr;
    return *this;
  }

  ~Dim() {
    delete[] _ptr;
    _ptr = nullptr;
  }

  int size(uint32_t idx) const { return _dims[idx]; }

  const int &at(size_t idx) const { return _dims.at(idx); }

  /**
   * @brief calculate size of allocated memory
   * 
   * @return size_t size of \c Dim::_ptr, in bytes.
   */
  size_t allocated_bytes() const {
    return _size * sizeof(T);
  }

  /**
   * @brief Checks if the Dim has no allocated memory and no dimension information
   * 
   * @return true if Dim has no allocated memory and no dimension information.
   * @return false otherwise
   */
  bool empty() const { return _dims.empty() and _size == 0 and _ptr == nullptr; }

private:
  /**
   * @brief calculate product of Dim::_dims
   * @return int
   */
  int _recalclulate_size() const {
    int product = 1;
    for (const int dim : _dims) {
      product *= dim;
    }
    return product;
  }

  /**
   * @brief update Dim::_size by Dim::_recalclulate_size and then re-allocate
   * memeory.
   */
  void _allocate() {
    _size = _recalclulate_size();
    if (_size > 0) {
      this->_ptr = new T[_size];
      std::fill_n(_ptr, _size, 0);
    } else {
      this->_ptr = nullptr;
    }
  }
};

} // namespace vision
#endif
