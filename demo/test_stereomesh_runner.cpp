#include <memory>                        // for shared_ptr
#include <string>                        // for string
#include <tuple>                         // for tie, ignore, tuple
#include <vector>                        // for vector

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utility.hpp>

#include "dlcv.h"
#include "stereomesh_runner.h"
#include "logging.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        return 1;
    }

    std::string   cfgfile          = argv[1];
    std::string   left_image_file  = argv[2];
    std::string   right_image_file = argv[3];
    
    cv::Mat       left_image       = cv::imread(left_image_file);
    cv::Mat       right_image      = cv::imread(right_image_file);
    if(true == left_image.empty())
    {
        LOG(ERROR) << "cannot read left image from: " << left_image_file;
        return 0;
    }
    if(true == right_image.empty())
    {
        LOG(ERROR) << "cannot read right image from: " << right_image_file;
        return 0;
    }

    auto stereomesh_runner   = new vision::StereoMeshRunner(cfgfile);
    auto output              = vision::StereoMeshRunner::output_type();
    std::vector<cv::Mat*>    images({&left_image, &right_image});
    stereomesh_runner->run(images, output);

    float*      out_data  = (float*)(output[0].data.lock().get());
    const int   h         = output[0].shape[1];
    const int   w         = output[0].shape[2];
    const int   c         = output[0].shape[3];
    // clean up
    delete stereomesh_runner;
    return 0;
}