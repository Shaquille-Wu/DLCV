#include <cstdint>              // for uint8_t
#include <memory>               // for shared_ptr
#include <tuple>                // for get
#include <typeinfo>             // for typeid
#include <typeindex>            // for type_index
#include <utility>              // for move
#include <vector>               // for vector

#include <gtest/gtest.h>

#include "blob.h"               // for Blob
#include "dlcv.h"               // for ElementType, F32

using namespace vision;

Blob create_ref() {
  // create a reference to test copy/move ctor
  auto ref = Blob::create<float>({2, 3, 5, 7});
  ref.for_each<float>([](float &value, size_t i) { value = float(i); });
  return ref;
}

TEST(Blob, ctor) {
  // default ctor -> empty, non-typed Blob
  auto default_blob = Blob();
  EXPECT_TRUE(default_blob.empty() and default_blob.use_count() == 0);

  // build empty, typed Blob
  auto float_blob = Blob(std::type_index(typeid(float)), sizeof(float));
  EXPECT_TRUE(float_blob.empty());
  EXPECT_TRUE(float_blob.contains<float>());
  EXPECT_EQ(float_blob.use_count(), 0);

  // build sized & typed Blob, copy dim
  auto dims = std::vector<int>{1, 3, 5, 7};
  auto blob = Blob(std::type_index(typeid(float)), sizeof(float), dims);
  EXPECT_EQ(blob.element_size(), 4);
  EXPECT_EQ(blob.num_elements(), 105);
  EXPECT_TRUE(blob.contains<float>());
  EXPECT_FALSE(dims.empty());

  // build sized & typed Blob, move dim
  blob = Blob(std::type_index(typeid(float)), sizeof(float), std::move(dims));
  EXPECT_EQ(blob.element_size(), 4);
  EXPECT_EQ(blob.num_elements(), 105);
  EXPECT_TRUE(blob.contains<float>());
  EXPECT_TRUE(dims.empty());

  // copy ctor
  auto ref = create_ref();
  const auto dup = ref;
  EXPECT_TRUE(ref == dup);
  EXPECT_TRUE(dup.use_count() == 1 and ref.use_count() == 1);

  // move ctor
  auto alisa = std::move(ref);
  EXPECT_FALSE(alisa.empty());
  EXPECT_TRUE(ref.empty());
  EXPECT_TRUE(alisa == create_ref());
  EXPECT_TRUE(alisa.use_count() == 1 and ref.use_count() == 0);
}

TEST(Blob, assign) {
  auto ref = create_ref();
  Blob slot, holder;

  // copy assign
  EXPECT_TRUE(slot.empty() and not ref.empty());
  EXPECT_TRUE(slot.use_count() == 0 and ref.use_count() == 1);
  slot = ref;
  EXPECT_TRUE(not slot.empty() and not ref.empty());
  EXPECT_TRUE(slot.use_count() == 1 and ref.use_count() == 1);
  EXPECT_TRUE(slot == ref);
  EXPECT_NE(slot.get<float>(), ref.get<float>());

  // move assign
  EXPECT_TRUE(holder.empty() and not ref.empty());
  EXPECT_TRUE(holder.use_count() == 0 and ref.use_count() == 1);
  const float * ptr = ref.get<float>();
  holder = std::move(ref);
  EXPECT_TRUE(not holder.empty() and ref.empty());
  EXPECT_TRUE(holder.use_count() == 1 and ref.use_count() == 0);
  EXPECT_TRUE(holder == create_ref());
  EXPECT_EQ(holder.get<float>(), ptr);
}

TEST(Blob, reset) {
  // reset an empty Blob
  auto empty = Blob();
  EXPECT_TRUE(empty.empty() and empty.use_count() == 0);
  empty.reset();
  EXPECT_TRUE(empty.empty() and empty.use_count() == 0);

  // reset a non-empty Blob
  auto ref = create_ref();
  EXPECT_TRUE(not ref.empty() and ref.use_count() == 1);
  ref.reset();
  EXPECT_TRUE(ref.empty() and ref.use_count() == 0);
}

TEST(Blob, create) {
  auto f32 = Blob::create<float>({1, 2, 4, 4});
  EXPECT_EQ(f32.num_elements(), 32);
  EXPECT_EQ(f32.element_size(), sizeof(float));
  EXPECT_EQ(f32.byte_size(), 32 * sizeof(float));

  auto dims = std::vector<int>{1, 3, 32, 32};
  auto u3k = Blob::create<uint8_t>(dims);
  EXPECT_EQ(u3k.size(0), 1);
  EXPECT_EQ(u3k.size(1), 3);
  EXPECT_EQ(u3k.size(2), 32);
  EXPECT_EQ(u3k.size(3), 32);
  EXPECT_EQ(u3k.num_elements(), 3072);
  EXPECT_EQ(u3k.element_size(), 1);
  EXPECT_EQ(u3k.byte_size(), 3072);
  EXPECT_FALSE(dims.empty());

  u3k = Blob::create<uint8_t>(std::move(dims));
  EXPECT_EQ(u3k.size(0), 1);
  EXPECT_EQ(u3k.size(1), 3);
  EXPECT_EQ(u3k.size(2), 32);
  EXPECT_EQ(u3k.size(3), 32);
  EXPECT_EQ(u3k.num_elements(), 3072);
  EXPECT_EQ(u3k.element_size(), 1);
  EXPECT_TRUE(dims.empty());
}

TEST(Blob, create_shared) {
  auto f32 = Blob::create_shared<float>({1, 2, 4, 4});
  EXPECT_EQ(f32->num_elements(), 32);
  EXPECT_EQ(f32->element_size(), sizeof(float));

  auto dims = std::vector<int>{1, 3, 32, 32};
  auto u3k = Blob::create_shared<uint8_t>(dims);
  EXPECT_EQ(u3k->size(0), 1);
  EXPECT_EQ(u3k->size(1), 3);
  EXPECT_EQ(u3k->size(2), 32);
  EXPECT_EQ(u3k->size(3), 32);
  EXPECT_EQ(u3k->num_elements(), 3072);
  EXPECT_EQ(u3k->element_size(), 1);
  EXPECT_FALSE(dims.empty());

  u3k = Blob::create_shared<uint8_t>(std::move(dims));
  EXPECT_EQ(u3k->size(0), 1);
  EXPECT_EQ(u3k->size(1), 3);
  EXPECT_EQ(u3k->size(2), 32);
  EXPECT_EQ(u3k->size(3), 32);
  EXPECT_EQ(u3k->num_elements(), 3072);
  EXPECT_EQ(u3k->element_size(), 1);
  EXPECT_TRUE(dims.empty());
}

TEST(Blob, contains) {
  auto f32 = Blob::create<float>();
  EXPECT_TRUE(f32.contains<float>());
  EXPECT_FALSE(f32.contains<uint8_t>());
}

TEST(Blob, at) {
  auto blob = create_ref();
  for (auto i = 0; i < blob.num_elements(); ++i) {
    EXPECT_EQ(blob.at<float>(i), i);
  }
}

TEST(Blob, size) {
  auto blob = create_ref();
  auto dims = std::vector<int>{2, 3, 5, 7};
  EXPECT_EQ(blob.size(0), 2);
  EXPECT_EQ(blob.size(1), 3);
  EXPECT_EQ(blob.size(2), 5);
  EXPECT_EQ(blob.size(3), 7);
  EXPECT_EQ(blob.size(), dims);
  EXPECT_EQ(blob.byte_size(), 2 * 3 * 5 * 7 * sizeof(int));
}

TEST(Blob, for_each) {
  auto blob = create_ref();
  blob.for_each<float>([](float &value) { value = size_t(value) % 233; });
  blob.for_each<float>(
      [](float &value, size_t i) { EXPECT_EQ(value, float(i % 233)); });
}

TEST(Blob, share_featuremap) {
  auto blob = create_ref();
  EXPECT_FALSE(blob.empty());
  auto feature_map = blob.share_featuremap<float>();
  auto dims = std::vector<int>{2, 3, 5, 7};
  // ensure blob is not changed
  EXPECT_FALSE(blob.empty());
  EXPECT_EQ(feature_map.shape, dims);
  EXPECT_EQ(feature_map.type, ElementType::F32);
  // ensure no memory copy
  EXPECT_FALSE(feature_map.data.expired());
  EXPECT_EQ(feature_map.data.lock().get(),
            reinterpret_cast<void *>(blob.get<float>()));
  EXPECT_EQ(blob.use_count(), 1);

  blob.reset();
  EXPECT_TRUE(feature_map.data.expired());
}

TEST(Blob, use_count) {
  EXPECT_EQ(Blob().use_count(), 0);
  const auto blob = create_ref();
  EXPECT_EQ(blob.use_count(), 1);
  auto shared_feature_map = blob.share_featuremap<float>();
  EXPECT_EQ(blob.use_count(), 1);
  auto dup = blob;
  EXPECT_EQ(blob.use_count(), 1);
  EXPECT_EQ(dup.use_count(), 1);
}

TEST(Blob, compare) {
  auto lhs = create_ref();
  auto rhs = create_ref();
  EXPECT_EQ(lhs, rhs);
  EXPECT_EQ(rhs, lhs);

  Blob e1, e2;
  EXPECT_EQ(e1, e2);
  EXPECT_NE(e1, lhs);
  EXPECT_NE(e2, lhs);
}
 
TEST(Blob, traits) {
  EXPECT_TRUE(std::is_nothrow_move_constructible<Blob>::value);
  EXPECT_TRUE(std::is_nothrow_move_assignable<Blob>::value);
}

TEST(Blob, vector_of_blobs) {
  // 1. create an empty blob
  std::vector<Blob> blobs;
  EXPECT_EQ(0, blobs.capacity());
  EXPECT_EQ(0, blobs.size());
  
  // 2. insert first Blob. alloc wont crash anything
  blobs.push_back(Blob::create<float>({1, 2, 4, 8}));
  EXPECT_EQ(1, blobs.capacity());
  EXPECT_EQ(1, blobs.size());
  float* buffer_0 = blobs[0].get<float>();
  
  // 3. insert 2nd Blob. 1st realloc
  blobs.push_back(Blob::create<float>({1, 2, 4, 8}));
  EXPECT_EQ(2, blobs.capacity());
  EXPECT_EQ(2, blobs.size());
  float* buffer_1 = blobs[1].get<float>();

  // 4. insert 3rd Blob. 2nd realloc
  blobs.push_back(Blob::create<float>({1, 2, 4, 8}));
  EXPECT_EQ(4, blobs.capacity());
  EXPECT_EQ(3, blobs.size());
  float* buffer_2 = blobs[2].get<float>();

  // 5. ensure all previously inserted blobs are MOVED
  //    to reallocated memory. if any of these expectations
  //    falls, Inference::Inference will segfalut when
  //    _indata.size() > 1 or _outdata.size() > 1
  EXPECT_EQ(buffer_0, (blobs[0].get<float>()));
  EXPECT_EQ(buffer_1, (blobs[1].get<float>()));
  EXPECT_EQ(buffer_2, (blobs[2].get<float>()));
}