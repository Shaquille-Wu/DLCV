#include <algorithm>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "dim.h"
#include "dim_helper.h"

using vision::Dim;

TEST(Dim, size) {
  const auto dim = Dim<float>{1, 3, 64, 64};
  EXPECT_EQ(dim._size, 1 * 3 * 64 * 64);
  EXPECT_EQ(dim.allocated_bytes(), 1 * 3 * 64 * 64 * 4);

  // _dims, size, at should give dimension as expect
  EXPECT_EQ(dim.size(0), 1);
  EXPECT_EQ(dim.size(1), 3);
  EXPECT_EQ(dim.size(2), 64);
  EXPECT_EQ(dim.size(3), 64);
  EXPECT_EQ(dim.at(0), 1);
  EXPECT_EQ(dim.at(0), 1);
  EXPECT_EQ(dim.at(0), 1);
  EXPECT_EQ(dim.at(0), 1);
}

TEST(Dim, CopyCtor) {
  const auto dim = Dim<float>{1, 3, 64, 64};
  std::generate_n(dim._ptr, dim._size, []() {
    static float value = 0;
    return value++;
  });

  // copy ctor
  const auto dup = Dim<decltype(dim)::value_type>(dim);
  EXPECT_NE(dim._ptr, dup._ptr);
  EXPECT_TRUE(dim_equal(dim, dup));
}

TEST(Dim, CopyAssign) {
  const auto dim = Dim<float>{1, 3, 64, 64};
  std::generate_n(dim._ptr, dim._size, []() {
    static float value = 0;
    return value++;
  });

  // copy assign
  auto empty = Dim<float>();
  EXPECT_TRUE(empty.empty());
  empty = dim;
  EXPECT_NE(empty._ptr, dim._ptr);
  EXPECT_TRUE((dim_equal(empty, dim)));
  EXPECT_FALSE(empty.empty());

  // copy assign
  auto slot = Dim<float>{1, 2, 3, 4};
  slot = dim;
  EXPECT_NE(slot._ptr, dim._ptr);
  EXPECT_TRUE((dim_equal(slot, dim)));
}

TEST(Dim, InitializerListCtor) {
  const int dim_array[4]{1, 3, 64, 64};
  const auto dim = Dim<float>(4, dim_array);
  const auto another = Dim<float>{1, 3, 64, 64};
  EXPECT_EQ(dim._dims, another._dims);
  EXPECT_EQ(dim._size, another._size);
}

TEST(Dim, MoveCtor) {
  auto value = Dim<float>{1, 2, 3, 4};
  auto ref = value._ptr;
  Dim<float> slot(std::move(value));

  const auto expected_dims = std::vector<int>{1, 2, 3, 4};
  EXPECT_EQ(slot._dims, expected_dims);
  EXPECT_EQ(slot._size, 1 * 2 * 3 * 4);
  EXPECT_EQ(slot._ptr, ref);

  EXPECT_TRUE(value._dims.empty());
  EXPECT_EQ(value._size, 0);
  EXPECT_EQ(value._ptr, nullptr);
}

TEST(Dim, MoveAssign) {
  auto slot = Dim<float>();
  auto value = Dim<float>{1, 2, 3, 4};
  auto ref = value._ptr;
  slot = std::forward<Dim<float>>(value);

  const auto expected_dims = std::vector<int>{1, 2, 3, 4};
  EXPECT_EQ(slot._dims, expected_dims);
  EXPECT_EQ(slot._size, 1 * 2 * 3 * 4);
  EXPECT_EQ(slot._ptr, ref);

  EXPECT_TRUE(value._dims.empty());
  EXPECT_EQ(value._size, 0);
  EXPECT_EQ(value._ptr, nullptr);
}

TEST(Dim, EmptyCheck) {
  Dim<float> dim;
  EXPECT_TRUE(dim.empty());
}