#include <memory>         // for unique_ptr
#include <string>         // for string
#include <utility>        // for move

#include <opencv2/opencv.hpp>

#include "dlcv.h"         // for DLCVOut
#include "modelrunner.h"  // for ModelRunner
#include "imodel_runner.h"    // for IModelRunner

namespace vision {
    class IModelRunner::Impl { //Hide inner implements from interface
        public:
          explicit Impl(std::string config_file, std::string jpatch):
              runner(std::move(config_file), std::move(jpatch)) {}

          int run(cv::Mat &image, vision::DLCVOut& out) {
              return runner.run(image, out);
          }
          int run_images(std::vector<cv::Mat*> &images, vision::DLCVOut& out) {
              return runner.run_images(images, out);
          }
        private:
          ModelRunner runner;
    }; // class IModelRunner::Impl

    IModelRunner::IModelRunner(std::string config_file, std::string jpatch){
        pimpl = new IModelRunner::Impl(std::move(config_file), std::move(jpatch));
        CHECK(pimpl != NULL);
    }

    int IModelRunner::run(cv::Mat &image, vision::DLCVOut& out) {
        return pimpl->run(image, out);
    }

    int IModelRunner::run_images(std::vector<cv::Mat*> &images, vision::DLCVOut& out) {
        return pimpl->run_images(images, out);
    }

    IModelRunner::~IModelRunner(){
        if (pimpl) {
            delete pimpl;
            pimpl = nullptr;
        }
    }

} // namespace vision
