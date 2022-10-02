#include <memory>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "modelrunner.h"
#include "keypoints_attributes.h"
#include "logging.h"

namespace vision {
    class KeypointsAttributes::Impl {
        public:
          explicit Impl(std::string config_file) : runner(std::move(config_file)) {}
          int run(cv::Mat &image, std::vector<cv::Point>& keypoints, std::vector<std::vector<float> >& attributes) {
              DLCVOut out;
              const auto rtv = runner.run(image, out);
              if (rtv != 0)
                  return rtv;
              if (out.has_keypoints) {
                  keypoints = out.keypoints;
              } else {
                  LOG(ERROR) << "Error: keypoints result is null";
                  return -1;
              }
              if (out.has_multiclass) {
                  attributes = out.multiclass;
              }
              return 0;
          }
        private:
          ModelRunner runner;
    }; // class KeypointsAttributes::Impl

    KeypointsAttributes::KeypointsAttributes(std::string config_file){
        pimpl = new KeypointsAttributes::Impl(std::move(config_file));
        CHECK(pimpl != NULL);
    }

    int KeypointsAttributes::run(cv::Mat &image, std::vector<cv::Point>& keypoints, std::vector<std::vector<float> >& attributes) {
        return pimpl->run(image, keypoints, attributes);
    }

    KeypointsAttributes::~KeypointsAttributes(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
