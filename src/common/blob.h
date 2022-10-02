#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include <cstring>
#include <utility>

#include "logging.h"
#include "dlcv.h"

namespace vision {

namespace blob_config {

/**
 * @brief compile-time min
 */
template <typename T>
constexpr T min(T lhs, T rhs)
{
  return lhs < rhs ? lhs : rhs;
}

using dim_type = int;
constexpr size_t minimum_dims = 4;
constexpr size_t maximum_dims =
    min(minimum_dims, sizeof(std::vector<dim_type>) / sizeof(dim_type));

using dims_type = std::vector<dim_type>;

} // namespace blob_config

template<ElementType Value>
using element_type_constant = std::integral_constant<ElementType, Value>;

/**
 * @brief mapping arithmetic type to vision::ElementType
 * 
 * @tparam T arithmetic type, must be one of: i8, u8, i16, u16, i32, u32, f32
 */
template<typename T> struct element_type_of;
template <> struct element_type_of<int8_t> : element_type_constant<ElementType::I8> {};
template <> struct element_type_of<uint8_t> : element_type_constant<ElementType::U8> {};
template <> struct element_type_of<int16_t> : element_type_constant<ElementType::I16> {};
template <> struct element_type_of<uint16_t> : element_type_constant<ElementType::U16> {};
template <> struct element_type_of<int32_t> : element_type_constant<ElementType::I32> {};
template <> struct element_type_of<uint32_t> : element_type_constant<ElementType::U32> {};
template <> struct element_type_of<float> : element_type_constant<ElementType::F32> {};

/**
 * @brief Blob - the successor of vision::Dim.
 * 
 * See \a test/test_blob.cpp for usage.
 */
class Blob {
 public:
  using dim_t = blob_config::dim_type;
  using dims_t = blob_config::dims_type;

  /**
   * @brief Construct a new, empty Blob object with unsettled type
   */
  Blob()
      : type_(std::type_index(typeid(void))),
        element_size_(0), dims_(), num_elements_(0), managed_(nullptr) {}

  /**
   * @brief Construct a new, typed, empty Blob object
   * 
   * @param type type_index of element
   * @param element_size element size in bytes
   */
  Blob(std::type_index type, size_t element_size)
      : type_(type), element_size_(element_size), dims_(), num_elements_(0), managed_(nullptr) {}

  /**
   * @brief Construct a new, typed, allocated Blob object. This ctor will allocate memory.
   * 
   * @param type type_index of element
   * @param element_size element size in bytes
   * @param dims dimension of blob
   */
  Blob(std::type_index type, size_t element_size, const dims_t &dims)
      : type_(type), element_size_(element_size), dims_(dims) {
    num_elements_ = recalclulate_size();
    allocate(num_elements_ * element_size_);
  }
  
  /**
   * @brief Construct a new, typed, allocated Blob object.
   *        This ctor will allocate memory.
   *        Same as previous, but use rvalue reference of \a dims
   * 
   * @param type type_index of element
   * @param element_size element size in bytes
   * @param dims rvalue of dimension.
   */
  Blob(std::type_index type, size_t element_size, dims_t &&dims)
      : type_(type), element_size_(element_size), dims_(std::forward<dims_t>(dims)) {
    num_elements_ = recalclulate_size();
    allocate(num_elements_ * element_size_);
  }

  /**
   * @brief Copy constructor of new Blob object
   * 
   * @param other blob object to copy
   */
  Blob(const Blob &other)
      : type_(other.type_), element_size_(other.element_size_),
        dims_(other.dims_), num_elements_(other.num_elements_) {
    copy_from_external(other.managed_.get());
  }

  /**
   * @brief Move constructor of Blob object
   * 
   * @param other rvalue of blob to move
   */
  Blob(Blob &&other) noexcept
      : type_(other.type_), element_size_(other.element_size_),
        dims_(std::move(other.dims_)), num_elements_(other.num_elements_) {
    managed_.reset();
    managed_.swap(other.managed_);
    other.reset();
  }

  /**
   * @brief Destroy the Blob object
   * 
   * This dtor just calls \a .reset() of managed pointer.
   */
  ~Blob() { managed_.reset(); }

  /**
   * @brief Copy assign operator of Blob
   * 
   * @param other \a Blob object to copy
   * @return Blob& reference of this blob
   */
  Blob &operator=(const Blob &other) {
    managed_.reset();
    type_ = other.type_;
    dims_ = other.dims_;
    num_elements_ = other.num_elements_;
    element_size_ = other.element_size_;
    copy_from_external(other.managed_.get());
    return *this;
  }

  /**
   * @brief Move assign operator of Blob
   * 
   * @param other \a Blob object to move
   * @return Blob& reference of this blob
   */
  Blob &operator=(Blob &&other) noexcept {
    managed_.reset();
    type_ = other.type_;
    dims_ = std::move(other.dims_);
    num_elements_ = other.num_elements_;
    element_size_ = other.element_size_;
    managed_.swap(other.managed_);
    other.reset();
    return *this;
  }

  /**
   * @brief Free previously allocated memeory (if any) and set size & dimension to zero,
   *        type information of element will be erased.
   */
  void reset() noexcept {
    // 
    type_ = std::type_index(typeid(void));
    // void is an incomplete type so sizeof(void) is ill-formed.
    // But its reasonable enough for non-initialized value anyway.
    element_size_ = 0;
    num_elements_ = 0;
    dims_.clear();
    managed_.reset();
  }

  /**
   * @brief Checks if the blob has no elements, i.e. whether \p num_elements() == 0.
   * 
   * @return true if the blob is empty
   * @return false otherwise
   */
  bool empty() const noexcept { return num_elements() == 0; }

  /**
   * @brief Returns the number of elements in the blob.
   * 
   * @return size_t The number of elements in the blob.
   */
  size_t num_elements() const noexcept { return num_elements_; }

  /**
   * @brief Returns the size of elements in the blob.
   * 
   * @return size_t The size of elements in the blob, in bytes.
   */
  size_t element_size() const noexcept { return element_size_; }

  /**
   * @brief Returns the size of blob, in bytes.
   * 
   * @return size_t The size of blob, in bytes.
   */
  size_t byte_size() const noexcept {
    return num_elements_ * element_size_;
  }

  /**
   * @brief Create a new, typed, empty Blob object
   * @sa Blob::Blob(std::type_index, size_t)
   * 
   * @tparam T element type
   * @return created Blob instance
   */
  template <typename T>
  static Blob create() {
    return Blob(std::type_index(typeid(T)), sizeof(T));
  }

  /**
   * @brief Create a new, typed, allocated Blob object
   * @sa Blob::Blob(std::type_index, size_t, const dims_t&)
   * 
   * @tparam T element type
   * @param dims dimenson of the Blob
   * @return created Blob instance
   */
  template <typename T>
  static Blob create(const dims_t &dims) {
    return Blob(std::type_index(typeid(T)), sizeof(T), dims);
  }
  
  /**
   * @brief Create a new, typed, allocated Blob object
   * @sa Blob::Blob(std::type_index, size_t, dims_t&&)
   * 
   * @tparam T element type
   * @param dims dimenson of the Blob
   * @return created Blob instance
   */
  template <typename T>
  static Blob create(dims_t &&dims) {
    return Blob(std::type_index(typeid(T)), sizeof(T), std::forward<dims_t>(dims));
  }

  /**
   * @brief Create a new, typed, empty, shared blob.
   * @sa Blob::create<T>()
   * 
   * @tparam T 
   * @return std::shared_ptr<Blob> 
   */
  template <typename T>
  static std::shared_ptr<Blob> create_shared() {
    return std::make_shared<Blob>(std::type_index(typeid(T)), sizeof(T));
  }

  /**
   * @brief Create a new, typed, allocated, shared blob.
   * @sa Blob::create<T>(const dims_t&)
   * 
   * @tparam T 
   * @param dims 
   * @return std::shared_ptr<Blob> 
   */
  template <typename T>
  static std::shared_ptr<Blob> create_shared(const dims_t &dims) {
    return std::make_shared<Blob>(std::type_index(typeid(T)), sizeof(T), dims);
  }

  /**
   * @brief Create a new, typed, allocated, shared blob. 
   * @sa Blob::create<T>(dims_t&&)
   * 
   * @tparam T 
   * @param dims 
   * @return std::shared_ptr<Blob> 
   */
  template <typename T>
  static std::shared_ptr<Blob> create_shared(dims_t&& dims) {
    return std::make_shared<Blob>(std::type_index(typeid(T)), sizeof(T), std::forward<dims_t>(dims));
  }

  /**
   * @brief check if current Blob object's data type is T
   *
   * @tparam T expected type
   * @return true if current Blob object's data type is T
   * @return false otherwise
   */
  template <typename T>
  bool contains() const {
    return std::type_index(typeid(T)) == type_;
  }

  /**
   * @brief Returns a reference to the element at specified location \p i, with bounds & type checking.
   * 
   * @tparam T desired element type
   * @param i position of the element to return 
   * @return T& Reference to the requested element.
   */
  template <typename T>
  T &at(int i) {
    type_check_in<T>("at()");
    return reinterpret_cast<T *>(managed_.get())[i];
  }

  /**
   * @brief Returns a const reference to the element at specified location \p i, with bounds & type checking.
   * 
   * @tparam T desired element type
   * @param i position of the element to return 
   * @return const T& Reference to the requested element.
   */
  template <typename T>
  const T &at(int i) const {
    type_check_in<T>("at() const");
    return reinterpret_cast<T *>(managed_.get())[i];
  }
  
  /**
   * @brief Returns the reference of dimension in the blob
   * 
   * @return const dims_t& const reference to dimension
   */
  const dims_t &size() const {
    return dims_;
  }

  /**
   * @brief Returns the size of at \p demension in the blob 
   * 
   * @param demonsion 
   * @return dim_t 
   */
  dim_t size(size_t demonsion) const {
    return dims_.at(demonsion);
  }

  /**
   * @brief Returns the managed memory, with type checking.
   * 
   * @warning \p get() method DOES NOT release the ownership of memory.
   *          NEVER apply \a delete[] to the returned pointer.
   * 
   * @tparam T desired element type
   * @return T* returns address of the managed memory.
   */
  template<typename T>
  T* get() const {
    type_check_in<T>("get<T>() const");
    return reinterpret_cast<T*>(managed_.get());
  }

  /**
   * @brief Runs the given functor \p fn over all elements.
   * 
   * @tparam T desired element type
   * @param fn unary functor to apply.
   * @return Blob& return reference of this Blob
   */
  template <typename T>
  Blob &for_each(const std::function<void(T &)> &fn) {
    const auto begin = get<T>();
    const auto end = begin + num_elements();
    std::for_each(begin, end, fn);
    return *this;
  }

  /**
   * @brief Runs the given functor \p fn over all elements.
   * 
   * @tparam T desired element type
   * @param fn functor to apply. The 1st argument is reference to element and the 2nd argument is the index of element.
   * @return Blob& return reference of this Blob
   */
  template <typename T>
  Blob &for_each(const std::function<void(T &, size_t)> &fn) {
    const auto buffer = get<T>();
    for (size_t i = 0; i < num_elements(); ++i) {
      fn(buffer[i], i);
    }
    return *this;
  }

  /**
   * @brief compare two Blob instances
   * 
   * @param other Blob instance to compare with
   * @return true if \p *this and \p other have same type, same dimension, and contain same data
   * @return false otherwise
   */
  bool operator==(const Blob &other) const {
    if (type_ == other.type_ and dims_ == other.dims_ and num_elements_ == other.num_elements_ and element_size_ == other.element_size_) {
      return std::memcmp(managed_.get(), other.managed_.get(), num_elements_ * element_size_) == 0;
    }
    return false;
  }

  /**
   * @brief compare two Blob instances
   * 
   * @param other Blob instance to compare with
   * @return true if \p *this and \p other have different type, dimension, or contain different data
   * @return false otherwise
   */
  bool operator!=(const Blob &other) const {
    return not (*this == other);
  }

  /**
   * @brief Share managed memory with a \p FeatureMap object \p without memory copy.
   * 
   * @tparam T desired element type
   * @return FeatureMap dimension, element type and pointer.
   */
  template<typename T>
  FeatureMap share_featuremap() const {
    type_check_in<T>("share_featuremap()");
    return {dims_, element_type_of<T>::value, managed_};
  }

  /**
   * @brief Returns how many objects own shared_ptr to current \p Blob 's managed memory.
   * 
   * @return 0 if \p this->empty(), 1 if no other object referring, N + 1 for N objects referring current \p Blob
   */
  long use_count() const noexcept {
    return managed_.use_count();
  }

private:
  /**
   * @brief helper function to do type  runtime.
   *
   * @tparam T
   * @param caller
   */
  template <typename T>
  const void type_check_in(const std::string caller) const {
    const auto desired = std::type_index(typeid(T));
    if (desired != type_) {
#if !defined(NDEBUG)
      LogMessage(base_filename(__FILE__), __LINE__, FATAL).stream()
          << "type check failed in vision::Blob::" << caller << ": "
          << "required " << std::type_index(desired).name()
          << ", but current Blob contains " << std::type_index(type_).name();
#endif
      ABORT() << "Blob: type check failed";
    }
  }

  /**
   * @brief calculate product of Blob::_dims
   * @return int
   */
  size_t recalclulate_size() const
  {
    size_t product = 1;
    for (const auto dim : dims_)
    {
      product *= dim;
    }
    return product;
  }

  /**
   * @brief update Blob::num_elements_ by Blob::calclulate_size and then
   * re-allocate memeory.
   */
  void allocate(const size_t size_in_bytes) {
    managed_.reset(new unsigned char[size_in_bytes], deleter);
    std::fill_n(managed_.get(), size_in_bytes, 0);
  }

  static void deleter(unsigned char *ptr) noexcept { delete[](ptr); }

  /**
   * @brief copy from external data source
   *
   * @param data
   */
  void copy_from_external(const unsigned char *data)
  {
    if (num_elements_ > 0 and data != nullptr)
    {
      size_t size_in_byte = num_elements_ * element_size_;
      managed_.reset(new unsigned char[size_in_byte]);
      std::copy_n(data, size_in_byte, managed_.get());
    }
  }

  //! type_index of buffered data
  std::type_index type_;
  //! size for a managed element, in bytes
  size_t element_size_;
  //! dimension information. sizeof(dims_) <= sizeof(std::vector<anytype>)
  dims_t dims_;
  //! number of managed elements
  size_t num_elements_;
  //! managed buffer
  std::shared_ptr<unsigned char> managed_;
};

} // namespace vision
