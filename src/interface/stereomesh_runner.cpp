#include <memory>         // for unique_ptr
#include <string>         // for string
#include <utility>        // for move

#include <opencv2/opencv.hpp>

#include "dlcv.h"         // for FeatureMap, DLCVOut
#include "modelrunner.h"  // for ModelRunner
#include "stereomesh_runner.h"    // for StereoMeshRunner

namespace vision {
    class StereoMeshRunner::Impl {
        public:
          explicit Impl(std::string config_file) : runner(std::move(config_file)) {}
          int run(std::vector<cv::Mat*> &images, std::vector<FeatureMap>& featuremaps) {
              DLCVOut out;
              const auto rtv = runner.run_images(images, out);
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
    }; // class StereoMeshRunner::Impl

    StereoMeshRunner::StereoMeshRunner(std::string config_file)
        : pimpl(new StereoMeshRunner::Impl(std::move(config_file))) {}

    int StereoMeshRunner::run(std::vector<cv::Mat*> &images, std::vector<FeatureMap>& featuremaps) {
        return pimpl->run(images, featuremaps);
    }

    StereoMeshRunner::~StereoMeshRunner(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
