#include <memory>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "modelrunner.h"
#include "reidfeature.h"
#include "logging.h"

namespace vision {
    class Feature::Impl {
        public:
          explicit Impl(std::string config_file) : runner(std::move(config_file)) {}
          int run(cv::Mat &image, std::vector<float>& feature) {
              DLCVOut out;
              const auto rtv = runner.run(image, out);
              if (rtv != 0)
                  return rtv;
              if (out.has_feature) {
                  feature = std::move(out.feature);
              } else {
                  LOG(ERROR) << "Error: feature result is null";
                  return -1;
              }
              return 0;
          }
        private:
          ModelRunner runner;
    }; // class Feature::Impl

    Feature::Feature(std::string config_file)
        : pimpl(new Feature::Impl(std::move(config_file))) {}

    int Feature::run(cv::Mat &image, std::vector<float>& feature) {
        return pimpl->run(image, feature);
    }

    Feature::~Feature(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
