#include <cstdint>              // for uint8_t
#include <string>               // for string, operator==, basic_string

#include "featuremap_post.h"
#include "blob.h"               // for Blob, Blob::dims_t
#include "dlcv.h"               // for DLCVOut, FeatureMap 
#include "error_code.h"         // for ERROR_CODE_OUT_OF_MEMORY

namespace vision {

template<typename T>
Blob create_blob_from_output_shape(const json& node) {
  if (node.contains("output_shape")) {
    return Blob::create<T>(node.at("output_shape").get<std::vector<int>>());
  } else {
    // just returns am empty blob.
    // recolor & hwc2chw will initialize it during first inference.
    return Blob::create<T>();
  }
}

FeaturemapPost::FeaturemapPost(const json &node) {
  // 0. config input blob
  if (node.find("input") != node.cend()) {
    source = option<string>(node.at("input").get<string>());
  } else {
    source = option<string>::none();
  }

  // 1. config hwc2chw
  if(node.find("hwc2chw") != node.cend()) {
    hwc2chw_ = node.at("hwc2chw").get<bool>();
  } else {
    hwc2chw_ = false;
  }

  // 2. config mode
  const std::string mode_literal = node.at("mode").get<std::string>();
  const bool has_colormap = node.find("colormap") != node.cend();
  if (mode_literal == "recolor" and has_colormap) {
    colormap.reset(new std::vector<std::array<float, 3>>);
    for (const auto &color : node.at("colormap")) {
      if (color.is_array()) {
        const auto bgr = color.get<std::vector<float>>();
        if (bgr.size() != 3) {
          mode = kUNKNOWN;
          return;
        }
        colormap->push_back({{bgr[0], bgr[1], bgr[2]}});
      }
    }
    mode = kRECOLOR;
  } else if (mode_literal == "shared") {
    mode = kSHARED;
  } else {
    mode = kUNKNOWN;
  }

  // 3. config output buffers.
  //    NOTE: if `output_shape` in JSON mismatch actual input,
  //          buffers will be re-allocate.
  //    NOTE: if `output_shape` is omitted, buffer_ will be empty.
  //    TODO: support other types
  if ((mode == kSHARED and hwc2chw_)) {
    buffer_ = create_blob_from_output_shape<float>(node);
  }
  if (mode == kRECOLOR) {
    // RECOLOR mode creates UINT8 output
    buffer_ = create_blob_from_output_shape<uint8_t>(node);
  }
}

void hwc_to_chw(const Blob &blob, Blob& output) {
  const auto dims = blob.size().size();
  const auto height = blob.size(dims - 3);
  const auto width = blob.size(dims - 2);
  const auto channel = blob.size(dims - 1);
  const std::vector<int> output_size{blob.size(0), channel, height, width};
  if (output.size() != output_size) {
    output = Blob::create<float>(output_size);
  }
  float *out_ptr = output.get<float>();
  float *in_ptr = blob.get<float>();

  cv::Mat tmp_image(height, width, CV_32FC(channel), in_ptr);
  std::vector<cv::Mat> data;
  for (int i = 0; i < channel; ++i) {
    cv::Mat mat(height, width, CV_32FC1, out_ptr + i * width * height);
    data.push_back(mat);
  }
  cv::split(tmp_image, data);
}

Blob& recolor(const Blob &blob,
             Blob& canvas,
             const std::vector<std::array<float, 3>> &colormap) {
  int N = blob.size(0), H = 0, W = 0;
  if (blob.size(1) == 1) {
    H = blob.size(2);
    W = blob.size(3);
  } else {
    H = blob.size(1);
    W = blob.size(2);
  }
  const std::vector<int> output_size{N, H, W, 3};

  // canvas has shape Nx3xHxW
  if (canvas.size() != output_size) {
    canvas = Blob::create<uint8_t>(output_size);
  }

  for (auto i = 0; i < blob.num_elements(); ++i) {
    const auto &color = colormap.at(int(blob.at<float>(i)));
    canvas.at<uint8_t>(i * 3 + 0) = color[0];
    canvas.at<uint8_t>(i * 3 + 1) = color[1];
    canvas.at<uint8_t>(i * 3 + 2) = color[2];
  }
  return canvas;
}

int FeaturemapPost::run(const vector<Blob> &indata, const PreprocInfo& info, vision::DLCVOut &out) {
  if (mode == kUNKNOWN) {
    return ERROR_CODE_CONFIG_FORMAT;
  }
  if (indata.size() != 1) {
    return ERROR_CODE_PARAM_SIZE;
  }
  if (indata.front().empty()) {
    return ERROR_CODE_PARAM_NULL;
  }

  const Blob& blob = indata.front();
  return execute(blob, out);
}

int FeaturemapPost::run(const vector<Blob> &indata, const vector<string> &names,
                        const PreprocInfo &info, vision::DLCVOut &out) {
  if (mode == kUNKNOWN) {
    return ERROR_CODE_CONFIG_FORMAT;
  }
  if (indata.size() != names.size()) {
    return ERROR_CODE_PARAM_SIZE;
  }
  if (indata.empty() or names.empty()) {
    return ERROR_CODE_PARAM_NULL;
  }

  const size_t blob_index = select(names);
  if (blob_index >= indata.size()) {
    return ERROR_CODE_PARAM_NULL;
  }

  const Blob& blob = indata[blob_index];
  if (blob.empty()) {
    return ERROR_CODE_PARAM_NULL;
  }
  return execute(blob, out);
}

int FeaturemapPost::execute(const Blob &blob, vision::DLCVOut &out) {
  if (mode == kSHARED) {
    if (hwc2chw_) {
      // ALLOCATE ONLY ONCE IN hwc_to_chw
      hwc_to_chw(blob, buffer_);
      out.featuremaps.emplace_back(buffer_.share_featuremap<float>());
    } else {
      // NO COPY
      out.featuremaps.emplace_back(blob.share_featuremap<float>());
    }
    out.has_featuremaps = true;
  }

  if (mode == kRECOLOR) {
    // 1. RECOLOR mode requires blob in shape [N, 1, H, W] or [N, H, W, 1]
    const size_t ndims = blob.size().size();
    if (not (ndims == 4 and (blob.size(1) == 1 or blob.size(3) == 1))) {
      return ERROR_CODE_SHAPE_MISMATCH;
    }

    // 2. colormap must be set
    if (not colormap) {
      return ERROR_CODE_PARAM_NULL;
    }

    recolor(blob, buffer_, *colormap);
    out.featuremaps.emplace_back(buffer_.share_featuremap<uint8_t>());
    out.has_featuremaps = true;
  }
  return ERROR_CODE_SUCCESS;
}

size_t FeaturemapPost::select(const vector<string> &names) const {
  if (names.empty() or source.is_none()) {
    return 0;
  }
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == source.as_ref()) {
      return i;
    }
  }
  return names.size();
}

} // namespace vision
