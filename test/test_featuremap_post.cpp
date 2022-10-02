#include <algorithm>            // for copy_n
#include <array>                // for array, array<>::const_iterator
#include <cstdint>              // for uint8_t
#include <memory>               // for allocator_traits<>::value_type, uniqu...
#include <tuple>                // for get, tie, tuple
#include <vector>               // for vector

#include <gtest/gtest.h>

#include "blob.h"               // for Blob
#include "dlcv.h"               // for DLCVOut, FeatureMap, ElementType, U8
#include "error_code.h"         // for ERROR_*
#include "json.hpp"             // for operator""_json
#include "preprocess.h"         // for PreprocInfo
#include "featuremap_post.h"    // for FeaturemapPost, kCOPY, kRECOLOR, kUNKNOWN

using namespace vision;

const auto ref = {1, 1, 4, 4};
const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 0, 1, 2, 3, 4, 5}};

TEST(FeaturemapPost, ConfigWithInput) {
  const auto param = "{"
                     "  \"mode\": \"shared\","
                     "  \"input\": \"source\""
                     "}"_json;
  auto sp = FeaturemapPost(param);
  EXPECT_FALSE(sp.source.is_none());
  EXPECT_TRUE(sp.source.is_some());
  EXPECT_EQ(sp.source.as_ref(), "source");
}

TEST(FeaturemapPost, ConfigWithoutInput) {
  const auto param = "{"
                     "  \"mode\": \"shared\""
                     "}"_json;
  auto sp = FeaturemapPost(param);
  EXPECT_TRUE(sp.source.is_none());
  EXPECT_FALSE(sp.source.is_some());
}

TEST(FeaturemapPost, Select) {
  const auto param = "{"
                     "  \"mode\": \"shared\""
                     "}"_json;
  auto sp = FeaturemapPost(param);
  const std::vector<std::string> sentence{
      "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
  for (size_t i = 0; i < sentence.size(); ++i) {
    sp.source = option<std::string>(sentence[i]);
    EXPECT_EQ(sp.select(sentence), i);
  }
  sp.source = option<std::string>("anything else");
  EXPECT_EQ(sp.select(sentence), sentence.size());
}

TEST(FeaturemapPost, SharedMode) {
  // create FeaturemapPost
  const auto param = "{"
                     "  \"mode\": \"shared\""
                     "}"_json;
  auto sp = FeaturemapPost(param);
  // create & initialize dim
  auto ref = Blob::create<float>({1, 1, 4, 4});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info = PreprocInfo();
  EXPECT_EQ(sp.mode, kSHARED);
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ(in.front().size(), out.featuremaps.front().shape);
  // ensure memory is NOT copied
  void *original = in.front().get<float>();
  void *copied = out.featuremaps.front().data.lock().get();
  EXPECT_EQ(original, copied);
  EXPECT_EQ(in.front().use_count(), 1);

  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);
  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(copied, secondaty_ptr);
  EXPECT_EQ(secondary.featuremaps.front().data.use_count(), 1);

  // run again with source
  DLCVOut third;
  vector<string> names{"blob"};
  sp.source = option<string>("blob");
  EXPECT_EQ(sp.run(in, names, info, third), ERROR_CODE_SUCCESS);
  EXPECT_EQ(third.featuremaps.size(), 1);
  // ensure using same memory
  const void *third_ptr = third.featuremaps.front().data.lock().get();
  EXPECT_EQ(copied, third_ptr);
  EXPECT_EQ(third.featuremaps.front().data.use_count(), 1);
}

TEST(FeaturemapPost, SharedModePrealloc) {
  // create FeaturemapPost
  const auto param = "{"
                     "  \"mode\": \"shared\","
                     "  \"output_shape\": [1, 1, 4, 4]"
                     "}"_json;
  auto sp = FeaturemapPost(param);
  // create & initialize dim
  auto ref = Blob::create<float>({1, 1, 4, 4});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info = PreprocInfo();
  EXPECT_EQ(sp.mode, kSHARED);
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ(in.front().size(), out.featuremaps.front().shape);

  // ensure NO prealloc
  EXPECT_TRUE(sp.buffer_.empty());

  // ensure memory is NOT copied
  void *original = in.front().get<float>();
  void *copied = out.featuremaps.front().data.lock().get();
  EXPECT_EQ(original, copied);
  EXPECT_EQ(in.front().use_count(), 1);

  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);

  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(copied, secondaty_ptr);
  EXPECT_EQ(secondary.featuremaps.front().data.use_count(), 1);
}

TEST(FeaturemapPost, SharedModeHWC) {
  // create FeaturemapPost
  const auto param = "{"
                     "  \"mode\": \"shared\","
                     "  \"hwc2chw\": true "
                     "}"_json;
  constexpr int N = 1, C = 4, H = 2, W = 2;
  auto sp = FeaturemapPost(param);
  // create & initialize dim
  auto ref = Blob::create<float>({N, H, W, C});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info = PreprocInfo();
  EXPECT_TRUE(sp.hwc2chw_);
  EXPECT_EQ(sp.mode, kSHARED);
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ((std::vector<int>{N, C, H, W}), out.featuremaps.front().shape);
  // ensure memory is copied
  const float *original = in.front().get<float>();
  const float *copied = reinterpret_cast<float *>(out.featuremaps.front().data.lock().get());
  EXPECT_NE(copied, original);
  for (auto n = 0; n < N; ++n) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < H; ++h) {
        for (auto w = 0; w < W; ++w) {
          EXPECT_EQ(copied[n * C * H * W + c * H * W + h * W + w],
                    original[n * H * W * C + h * W * C + w * C + c]);
        }
      }
    }
  }
  
  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);
  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(copied, secondaty_ptr);
}

TEST(FeaturemapPost, SharedModeHWCPrealloc) {
  // create FeaturemapPost
  const auto param = "{"
                      "  \"mode\": \"shared\","
                      "  \"hwc2chw\": true,"
                      "  \"output_shape\": [1, 4, 2, 2]"
                      "}"_json;
  constexpr int N = 1, C = 4, H = 2, W = 2;
  auto sp = FeaturemapPost(param);
  // create & initialize dim
  auto ref = Blob::create<float>({N, H, W, C});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // ensure buffer allocated
  EXPECT_FALSE(sp.buffer_.empty());
  EXPECT_EQ(sp.buffer_.size(), (std::vector<int>{N, C, H, W}));
  const void* preallocated = sp.buffer_.get<float>();
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info = PreprocInfo();
  EXPECT_TRUE(sp.hwc2chw_);
  EXPECT_EQ(sp.mode, kSHARED);
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ((std::vector<int>{N, C, H, W}), out.featuremaps.front().shape);
  // ensure memory is copied
  const float *original = in.front().get<float>();
  const float *copied = reinterpret_cast<float *>(out.featuremaps.front().data.lock().get());
  EXPECT_NE(copied, original);
  EXPECT_EQ(copied, preallocated);
  for (auto n = 0; n < N; ++n) {
    for (auto c = 0; c < C; ++c) {
      for (auto h = 0; h < H; ++h) {
        for (auto w = 0; w < W; ++w) {
          EXPECT_EQ(copied[n * C * H * W + c * H * W + h * W + w],
                    original[n * H * W * C + h * W * C + w * C + c]);
        }
      }
    }
  }
  
  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);

  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(preallocated, secondaty_ptr);
}

TEST(FeaturemapPost, RecolorMode) {
  // create FeaturemapPost
  const auto param = "{"
                     "  \"mode\": \"recolor\","
                     "  \"colormap\": [[0, 0, 0],"
                     "                 [0, 128, 128],"
                     "                 [0, 0, 128],"
                     "                 [0, 128, 0],"
                     "                 [128, 0, 0],"
                     "                 [0, 0, 64],"
                     "                 [0, 0, 192],"
                     "                 [128, 128, 128],"
                     "                 [128, 128, 0],"
                     "                 [128, 0, 128]]"
                     "}"_json;
  std::vector<std::array<float, 3>> colormap{
      {{0, 0, 0}},  {{0, 128, 128}}, {{0, 0, 128}},     {{0, 128, 0}},   {{128, 0, 0}},
      {{0, 0, 64}}, {{0, 0, 192}},   {{128, 128, 128}}, {{128, 128, 0}}, {{128, 0, 128}}};
  auto sp = FeaturemapPost(param);
  // check colormap
  EXPECT_EQ(sp.mode, kRECOLOR);
  EXPECT_NE(sp.colormap, nullptr);
  EXPECT_EQ((*sp.colormap), colormap);
  // create & initialize Dim
  auto ref = Blob::create<float>({1, 1, 4, 4});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info= PreprocInfo();
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ(out.featuremaps.size(), 1);
  std::vector<std::vector<uint8_t>> pixels{
      {0, 0, 0},     {0, 128, 128}, {0, 0, 128}, {0, 128, 0},
      {128, 0, 0},   {0, 0, 64},    {0, 0, 192}, {128, 128, 128},
      {128, 128, 0}, {128, 0, 128}, {0, 0, 0},   {0, 128, 128},
      {0, 0, 128},   {0, 128, 0},   {128, 0, 0}, {0, 0, 64}};
  std::vector<int> shape;
  auto& feature_map = out.featuremaps.front();

  const auto expected_shape = std::vector<int>{1, 4, 4, 3};
  EXPECT_EQ(feature_map.shape, expected_shape);
  EXPECT_EQ(feature_map.type, ElementType::U8);
  bool success = true;
  auto ptr = reinterpret_cast<uint8_t*>(feature_map.data.lock().get());
  for (auto i = 0; i < 16; ++i) {
    success &= ptr[i * 3 + 0] == pixels[i][0] and
               ptr[i * 3 + 1] == pixels[i][1] and
               ptr[i * 3 + 2] == pixels[i][2];
  }
  EXPECT_TRUE(success);

  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);
  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(ptr, secondaty_ptr);
}

TEST(FeaturemapPost, RecolorModeHWC) {
  // create FeaturemapPost
  const auto param = "{"
                     "  \"mode\": \"recolor\","
                     "  \"hwc2chw\": true,"
                     "  \"colormap\": [[0, 0, 0],"
                     "                 [0, 128, 128],"
                     "                 [0, 0, 128],"
                     "                 [0, 128, 0],"
                     "                 [128, 0, 0],"
                     "                 [0, 0, 64],"
                     "                 [0, 0, 192],"
                     "                 [128, 128, 128],"
                     "                 [128, 128, 0],"
                     "                 [128, 0, 128]]"
                     "}"_json;
  std::vector<std::array<float, 3>> colormap{
      {{0, 0, 0}},  {{0, 128, 128}}, {{0, 0, 128}},     {{0, 128, 0}},   {{128, 0, 0}},
      {{0, 0, 64}}, {{0, 0, 192}},   {{128, 128, 128}}, {{128, 128, 0}}, {{128, 0, 128}}};
  auto sp = FeaturemapPost(param);
  // check colormap
  EXPECT_EQ(sp.mode, kRECOLOR);
  EXPECT_TRUE(sp.hwc2chw_);
  EXPECT_NE(sp.colormap, nullptr);
  EXPECT_EQ((*sp.colormap), colormap);
  // create & initialize Dim
  auto ref = Blob::create<float>({1, 4, 4, 1});
  const std::array<float, 16> values{{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 0, 1, 2, 3, 4, 5}};
  std::copy_n(values.cbegin(), 16, ref.get<float>());
  // run FeaturemapPost & check result
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info= PreprocInfo();
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_SUCCESS);
  EXPECT_EQ(out.featuremaps.size(), 1);
  std::vector<std::vector<uint8_t>> pixels{
      {0, 0, 0},     {0, 128, 128}, {0, 0, 128}, {0, 128, 0},
      {128, 0, 0},   {0, 0, 64},    {0, 0, 192}, {128, 128, 128},
      {128, 128, 0}, {128, 0, 128}, {0, 0, 0},   {0, 128, 128},
      {0, 0, 128},   {0, 128, 0},   {128, 0, 0}, {0, 0, 64}};
  std::vector<int> shape;
  auto& feature_map = out.featuremaps.front();

  const auto expected_shape = std::vector<int>{1, 4, 4, 3};
  EXPECT_EQ(feature_map.shape, expected_shape);
  EXPECT_EQ(feature_map.type, ElementType::U8);
  bool success = true;
  auto ptr = reinterpret_cast<uint8_t*>(feature_map.data.lock().get());
  for (auto i = 0; i < 16; ++i) {
    success &= ptr[i * 3 + 0] == pixels[i][0] and
               ptr[i * 3 + 1] == pixels[i][1] and
               ptr[i * 3 + 2] == pixels[i][2];
  }
  EXPECT_TRUE(success);

  // run twice
  DLCVOut secondary;
  EXPECT_EQ(sp.run(in, info, secondary), ERROR_CODE_SUCCESS);
  EXPECT_EQ(secondary.featuremaps.size(), 1);
  // ensure using same memory
  const void * secondaty_ptr = secondary.featuremaps.front().data.lock().get();
  EXPECT_EQ(ptr, secondaty_ptr);
}

TEST(FeaturemapPost, UnknownMode) {
  // create FeaturemapPost
  const auto param = "{ "
                     "  \"mode\": \"something-else\""
                     "} "_json;
  auto sp = FeaturemapPost(param);
  auto ref = Blob::create<float>({1, 1, 4, 4});
  auto in = std::vector<Blob>{ref};
  auto out = DLCVOut();
  auto info= PreprocInfo();
  EXPECT_EQ(sp.mode, kUNKNOWN);
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_CONFIG_FORMAT);
}

TEST(FeaturemapPost, ParamNull) {
  const auto param = "{ "
                     "  \"mode\": \"shared\""
                     "} "_json;
  auto sp = FeaturemapPost(param);
  auto in = std::vector<Blob>{};
  auto out = DLCVOut();
  auto info= PreprocInfo();
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_PARAM_SIZE);
  in.emplace_back();
  EXPECT_EQ(sp.run(in, info, out), ERROR_CODE_PARAM_NULL);
}

TEST(FeaturemapPost, MultipleInputs) {
  const auto param = "{ "
                     "  \"mode\": \"shared\","
                     "  \"input\": \"source\""
                     "} "_json;
  auto sp = FeaturemapPost(param);
  auto in = std::vector<Blob>{};
  auto names = std::vector<std::string>{"ignore", "source"};
  auto out = DLCVOut();
  auto info= PreprocInfo();
  EXPECT_EQ(sp.run(in, names, info, out), ERROR_CODE_PARAM_SIZE);
  in.emplace_back();
  EXPECT_EQ(sp.run(in, names, info, out), ERROR_CODE_PARAM_SIZE);
  in.emplace_back(Blob::create<float>({1, 5, 32, 32}));
  EXPECT_EQ(sp.run(in, names, info, out), ERROR_CODE_SUCCESS);
  EXPECT_TRUE(out.has_featuremaps);
  float *shared = reinterpret_cast<float *>(out.featuremaps.front().data.lock().get());
  float *expected = in.back().get<float>();
  EXPECT_EQ(expected, shared);
}