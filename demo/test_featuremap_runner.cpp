#include <memory>                        // for shared_ptr
#include <string>                        // for string
#include <tuple>                         // for tie, ignore, tuple
#include <vector>                        // for vector

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utility.hpp>
namespace fs = cv::utils::fs;

#include "dlcv.h"
#include "featuremap_runner.h"
#include "logging.h"

using vision::ElementType;

cv::Mat run(vision::FeaturemapRunner *runner, cv::Mat &image) {
  std::vector<vision::FeatureMap> output;
  runner->run(image, output);
  // FeaturemapPost in recolor mode outputs NHW3 frames
  auto &feature_map = output.front();
  const int height = feature_map.shape[1];
  const int width = feature_map.shape[2];
  cv::Mat prediction;
  if (feature_map.data.expired()) {
    LOG(ERROR) << "feature_map.data is expired";
    return prediction;
  }
  unsigned char *ptr = feature_map.data.lock().get();
  switch (feature_map.type) {
    case ElementType::F32:
      prediction = cv::Mat(height, width, CV_32FC1,
                           reinterpret_cast<float *>(ptr));
      break;
    case ElementType::U8:
      prediction = cv::Mat(height, width, CV_8UC3,
                           reinterpret_cast<unsigned char *>(ptr));
      break;
    default:
      LOG(ERROR) << "unsupported element type";
      break;
  }
  return prediction;
}

int main(int argc, char *argv[]) {
  // parse command line arguments
  const cv::String keys = 
    "{help h usage ? |      | print this message}"
    "{@model         |<none>| model (.json) to run}"
    "{@input         |<none>| input filename or path to directory containing input images}"
    "{@output        |<none>| output filename or directory to save results}"
    "{pattern        |*.png | pattern to glob images from input directory}";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Featuremap Runner CIL for DLCV " + vision::get_dlcv_version_string());

  // print usage / help message
  if (parser.has("help") or argc == 1) {
    parser.printMessage();
    return 0;
  }

  // exit on parse errors
  if (parser.check()) {
    parser.printErrors();
    return -1;
  }

  // extract command line arguments
  const cv::String model = parser.get<cv::String>("@model");
  const cv::String input = parser.get<cv::String>("@input");
  const cv::String output = parser.get<cv::String>("@output");
  const cv::String pattern = parser.get<cv::String>("pattern");

  // prepare input/output list
  std::vector<cv::String> source_files;
  std::vector<cv::String> output_files;
  if (fs::isDirectory(input)) {
    std::vector<cv::String> entries;
    fs::glob_relative(input, pattern, entries, true);
    for (const auto &entry : entries) {
      source_files.emplace_back(fs::join(input, entry));
      output_files.emplace_back(fs::join(output, entry));
    }
    if (fs::exists(output) and not fs::isDirectory(output)) {
      LOG(ERROR) << "Cannot create " << output << ": file exists";
    } else {
      fs::createDirectories(output);
    }
  } else {
    source_files.emplace_back(input);
    output_files.emplace_back(output);
  }

  // init runner
  auto featuremap_runner = new vision::FeaturemapRunner(model);
  
  // main-loop
  const size_t total = std::min(source_files.size(), output_files.size());
  for (size_t i = 0; i < total; ++i) {
    const auto& source_file = source_files[i];
    const auto& output_file = output_files[i];
    LOG(INFO) << "Processing image [" << 1 + i << " / " << total << "]: " << source_file << " -> " << output_file;
    cv::Mat image = cv::imread(source_file);
    const cv::Mat prediction = run(featuremap_runner, image);
    if (not prediction.empty()) {
      cv::imwrite(output_file, prediction);
    } else {
      LOG(ERROR) << "Failed processing: " << source_file;
    }
  }

  // clean up
  delete featuremap_runner;
  return 0;
}