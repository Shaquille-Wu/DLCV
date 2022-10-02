#ifndef _MODELRUNNER_H_
#define _MODELRUNNER_H_

#include<iostream>
#include "preprocess.h"
#include "postprocess.h"
#include "inference_proxy.h"

using namespace std;
using namespace nlohmann; // json

namespace vision{    
    
class ModelRunner {
public:
    ModelRunner(const string cfgfile, const string jpatch="");
    int run(cv::Mat& img, vision::DLCVOut& out);
    int run_images(std::vector<cv::Mat*>& imgs, vision::DLCVOut& out);
    ~ModelRunner();

    

private:
    std::vector<Preprocessor*>  _preprocessor;
    Inference*                  _inf;
    Postprocessor*              _postprocessor;
};

}
#endif
