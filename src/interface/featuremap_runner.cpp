#include <memory>         // for unique_ptr
#include <string>         // for string
#include <utility>        // for move

#include <opencv2/opencv.hpp>

#include "dlcv.h"         // for FeatureMap, DLCVOut
#include "modelrunner.h"  // for ModelRunner
#include "featuremap_runner.h"    // for FeaturemapRunner

namespace vision {
    class FeaturemapRunner::Impl {
        public:
          explicit Impl(std::string config_file) : runner(std::move(config_file)) {}
          int run(cv::Mat &image, std::vector<FeatureMap>& featuremaps) {
              DLCVOut out;
              const auto rtv = runner.run(image, out);
              if (rtv != 0)
                  return rtv;
              if (out.has_featuremaps) {
                  featuremaps = std::move(out.featuremaps);
              } else {
                  LOG(ERROR) << "Error: featuremap result is null";
                  return -1;
              }
              return 0;
          }
        private:
          ModelRunner runner;
    }; // class FeaturemapRunner::Impl

    FeaturemapRunner::FeaturemapRunner(std::string config_file)
        : pimpl(new FeaturemapRunner::Impl(std::move(config_file))) {}

    int FeaturemapRunner::run(cv::Mat &image, std::vector<FeatureMap>& featuremaps) {
        return pimpl->run(image, featuremaps);
    }

    FeaturemapRunner::~FeaturemapRunner(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
