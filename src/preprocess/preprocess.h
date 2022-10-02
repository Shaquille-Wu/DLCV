#ifndef _PREPROCESS_H_
#define _PREPROCESS_H_

#include<iostream>
#include "json.hpp"
#include "blob.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace nlohmann; //json

namespace vision{

// PreprocInfo
struct PreprocInfo {
    int origin_w;
    int origin_h;
    int curr_w; // current_w
    int curr_h;
    bool has_pad;
    int padding[4];
};

// PreprocOp
class PreprocOp {
public:
    PreprocOp();
    virtual ~PreprocOp(){};
    virtual int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out) = 0;
    bool _debug;
};


// ResizeOp
class ResizeOp : public PreprocOp {
    int _w, _h;
    int _mode;
public:
    ResizeOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// MeanFileOp
class MeanFileOp : public PreprocOp {
    int _w, _h, _c;
    std::vector<float> _means;
public:
    MeanFileOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// SwapChannelOp, RGB2BGR
class SwapChannelOp : public PreprocOp {
public:
    SwapChannelOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// ConvertDataTypeOp
class ConvertDataTypeOp : public PreprocOp {
    unsigned char _dtype;
public:
    ConvertDataTypeOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// ToGrayOp
class ToGrayOp : public PreprocOp {
    std::string _code;
public:
    ToGrayOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// GrayToBGROp
class GrayToBGROp : public PreprocOp {
    std::string _code;
public:
    GrayToBGROp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// BatchNormOp
class BatchNormOp : public PreprocOp {
    float _eps;
    vector<float> _mean;
    vector<float> _var;
    vector<float> _gamma;
    vector<float> _beta;
public:
    BatchNormOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

class EqualizeHistOp : public PreprocOp {
public:
    EqualizeHistOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// NormalizeOp out = (x - mean)/std
class NormalizeOp : public PreprocOp {
    vector<float> _mean;
    vector<float> _std;
public:
    NormalizeOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

class QuantizeOp : public PreprocOp {
    vector<float> _zero;
    vector<float> _step;
public:
    QuantizeOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, cv::Mat& out);
};

// totensor is different from PreprocOp
class ToTensorOp {
    bool _hwc2chw;
    bool _swapchannel;
    string _dtype;
public:
    ToTensorOp(const json& param);
    int run(cv::Mat& image, PreprocInfo& info, Blob &blob);
    bool _debug;
};

/*************************************************/

// function to make ops 
template<typename T> 
PreprocOp* make_preproc(const json& param) { 
    return new T(param); 
}

// Preprocessor
class Preprocessor {
    vector<PreprocOp* > _preproc_ops;
    ToTensorOp* _to_tensor;
    bool _debug;
    std::vector<cv::Mat>   _image_chain;
public:
    Preprocessor(const json& cfg);
    ~Preprocessor();
    int run(cv::Mat& image, PreprocInfo& info, Blob& outdata);
};


}
#endif
