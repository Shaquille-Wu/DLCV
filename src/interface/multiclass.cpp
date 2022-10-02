#include <memory>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "modelrunner.h"
#include "multiclass.h"
#include "logging.h"

namespace vision {
    class Multiclass::Impl {
        public:
          explicit Impl(std::string config_file) : runner(std::move(config_file)) {}
          int run(cv::Mat &image, std::vector<std::vector<float> >& multiclass) {
              DLCVOut out;
              const auto rtv = runner.run(image, out);
              if (rtv != 0)
                  return rtv;
              if (out.has_multiclass) {
                  multiclass = out.multiclass;
              } else {
                  LOG(ERROR) << "Error: multiclass result is null";
                  return -1;
              }
              return 0;
          }
        private:
          ModelRunner runner;
    }; // class Multiclass::Impl

    Multiclass::Multiclass(std::string config_file)
        : pimpl(new Multiclass::Impl(std::move(config_file))) {}

    int Multiclass::run(cv::Mat &image, std::vector<std::vector<float> >& multiclass) {
        return pimpl->run(image, multiclass);
    }

    Multiclass::~Multiclass(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
