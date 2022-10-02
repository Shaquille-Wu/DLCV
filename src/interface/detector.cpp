
#include "detector.h"
#include "modelrunner.h"
#include "dlcv.h"
#include "logging.h"

using namespace std;

namespace vision{


class Detector::Impl{ //Hide inner implements from interface
private:
    ModelRunner* _modelrunner;

public:
    Impl(const string cfgfile){
        _modelrunner = new ModelRunner(cfgfile); 
        CHECK(_modelrunner != NULL);
    }
    ~Impl(){
        if (_modelrunner){
            delete _modelrunner;
            _modelrunner = nullptr;
        }
    }

    int run(cv::Mat& image, vision::DLCVOut& out){
        return _modelrunner->run(image, out);
    } 

};

Detector::Detector(const string cfgfile){
    _pimpl = new Impl(cfgfile);    
    CHECK(_pimpl != NULL);
}

int Detector::run(cv::Mat& image, map<string, vector<Box> >& boxes_map, int frameid){
    DLCVOut out;
    int r = _pimpl->run(image, out);
    if (r != 0)
        return r;
    if (out.has_boxes)
        boxes_map = out.boxes_map;
    else{
        LOG(ERROR) << "Error: detector result is null";
        return -1;
    }
    return 0;
}

Detector::~Detector(){
    if (_pimpl){
        delete _pimpl;
        _pimpl = nullptr;
    }
}


}
