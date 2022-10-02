#include <cstring>
#include <ostream>
#include <fstream>

#include "json.hpp"
#include "error_code.h"
#include "preprocess.h"
#include "util.h"
#include "logging.h"
#include "../../dlcv_proc_opt/dlcv_proc_wrapper.h"


using namespace std;
using json=nlohmann::json;

namespace vision {

// PreprocOp
PreprocOp::PreprocOp() {
    _debug = false;
}

// ResizeOp
ResizeOp::ResizeOp(const json& param) {
    _w = param.at("width");
    _h = param.at("height");
    //_mode = param["interpolation"];
}
int ResizeOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    // set preproc info
    info.origin_w = image.cols;
    info.origin_h = image.rows;

    cv::resize(image, out, cv::Size(_w, _h), 0., 0, cv::INTER_LINEAR);
    if(_debug){
        cc_print_mat("resize", out);
    }
    return 0;
}

// MeanFileOp
MeanFileOp::MeanFileOp(const json& param) {
    _w = param.at("width");
    _h = param.at("height");
    _c = param.at("channel");
    std::string file = param.at("meanfile");
    file = get_abs_path(file);

    _means.resize(_w * _h * _c);

    std::ifstream ifs(file.c_str(), std::ios::binary | std::ios::in);
    CHECK(ifs.is_open() == true);
    ifs.read((char*)(&_means[0]), sizeof(float) * _w * _h * _c);
    ifs.close();
}
int MeanFileOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    CHECK(_c == image.channels());

    int rtype = CV_32FC3;
    switch (_c) {
        case 1:
          rtype = CV_32FC1;
          break;
        case 3:
          rtype = CV_32FC3;
          break;
        default:
          ABORT();
    }

    cv::Mat mean_mat(_h, _w, rtype, (float*)(&_means[0]));
    if(_w != image.cols || _h != image.rows) {
        cv::resize(mean_mat, mean_mat, cv::Size(image.cols, image.rows));
    }

    image.convertTo(image, CV_32F);
    out = image - mean_mat;

    return 0;
}

// SwapChannelOp
SwapChannelOp::SwapChannelOp(const json& param){
    LOG(INFO) << "Better to use swapchannel in totensor for better performance";
}
int SwapChannelOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    cv::cvtColor(image, out, cv::COLOR_RGB2BGR); 
    if(_debug){
        cc_print_mat("swapchannel", out);
    }
    return 0;
}

// ConvertDataTypeOp
ConvertDataTypeOp::ConvertDataTypeOp(const json& param){
    string dt = param.at("dtype"); 
    if ( dt == "float32") 
        _dtype = CV_32F;
    else{
        LOG(ERROR) << "Error: dtype not supported: " <<  dt;
        ABORT();
    }
}
int ConvertDataTypeOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    image.convertTo(out, _dtype);
    if(_debug){
        cc_print_mat("convert_dtype", out);
    }
    return 0;
} 

// ToGrayOp
ToGrayOp::ToGrayOp(const json& param) {
    _code = param.at("code");
    CHECK((_code == "bgr2gray" || _code == "rgb2gray"));
}
int ToGrayOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out) {
    if(image.channels() == 1) {
        out = image.clone();
        return 0;
    }

    proc_opt::dlcv_image_togray(image, out, "bgr2gray" == _code ? 0 : 1);
    if (_debug) {
        cc_print_mat("togray", out);
    }
    return 0;
}

// GrayToBGROp
GrayToBGROp::GrayToBGROp(const json& param) {
}
int GrayToBGROp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out) {
    if(image.channels() == 3) {
        out = image.clone();
        return 0;
    }
    cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);

    if (_debug) {
        cc_print_mat("tobgr", out);
    }
    return 0;
}


// BatchNormOp
BatchNormOp::BatchNormOp(const json& param){
    _eps = 1e-5;
    if (param.contains("eps") )
        _eps = param.at("eps");

    json beta = param.at("beta");
    for (json::iterator it = beta.begin(); it != beta.end(); ++it) {
        _beta.push_back(*it);
    }

    json mean = param.at("mean");
    for (json::iterator it = mean.begin(); it != mean.end(); ++it) {
        _mean.push_back(*it);
    }

    json gamma = param.at("gamma");
    for (json::iterator it = gamma.begin(); it != gamma.end(); ++it) {
        _gamma.push_back(*it);
    }

    json var = param.at("var");
    for (json::iterator it = var.begin(); it != var.end(); ++it) {
        _var.push_back(*it);
    }

    CHECK(_beta.size() == _mean.size());
    CHECK(_mean.size() == _gamma.size());
    CHECK(_gamma.size() == _var.size());
    CHECK(_beta.size() == 1 || _beta.size() == 3);

}
int BatchNormOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    int channels = image.channels();
    //bn: (x - mean) * gamma + beta
    //normalize: (x - mean)/std
    //so, we apply normalize to replace bn
    //new_mean = (mean - beta/gamma)
    //new_std  = 1/gamma
    std::vector<float>   norm_mean(channels, 0.0);
    std::vector<float>   norm_std(channels, 0.0);
    if (1 == channels) {
        cv::Scalar m_bn_mean  = cv::Scalar(_mean[0]);
        cv::Scalar m_bn_var   = cv::Scalar(1 / sqrt(_var[0] + _eps));
        cv::Scalar m_bn_gamma = cv::Scalar(_gamma[0]);
        cv::Scalar m_bn_beta  = cv::Scalar(_beta[0]);
        m_bn_gamma = m_bn_gamma * m_bn_var;

        norm_std[0]   = 1.0 / m_bn_gamma[0];
        norm_mean[0]  = m_bn_mean[0] - m_bn_beta[0] / m_bn_gamma[0];
    } else {
        assert(image.channels() == 3);

        cv::Scalar m_bn_mean  = cv::Scalar(_mean[0], _mean[1], _mean[2]);
        cv::Scalar m_bn_beta  = cv::Scalar(_beta[0], _beta[1], _beta[2]);
        cv::Scalar m_bn_gamma = cv::Scalar(_gamma[0] / (sqrt(_var[0] + _eps)),
                                           _gamma[1] / (sqrt(_var[1] + _eps)),
                                           _gamma[2] / (sqrt(_var[2] + _eps)));

        for(int i = 0 ; i < 3 ; i ++)
        {
            norm_std[i]   = 1.0 / m_bn_gamma[i];
            norm_mean[i]  = m_bn_mean[i] - m_bn_beta[i] / m_bn_gamma[i];
        }
    }

    proc_opt::dlcv_normalize_image(image, out, norm_mean, norm_std);

    if(_debug){
        cc_print_mat("batchnorm", out);
    }
    return 0;
}

// NormalizeOp out = (x - mean)/std
NormalizeOp::NormalizeOp(const json& param){
    json m = param.at("mean");
    for (json::iterator it = m.begin(); it != m.end(); ++it) {
        _mean.push_back(*it);
    }
    json std = param.at("std");
    for (json::iterator it = std.begin(); it != std.end(); ++it) {
        _std.push_back(*it);
    }
    CHECK (_mean.size() == _std.size());
    CHECK (_mean.size() == 1 || _mean.size() == 3);
}

int NormalizeOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out){
    // out = (x - mean)/std
    proc_opt::dlcv_normalize_image(image, out, _mean, _std);
    if(_debug){
        cc_print_mat("normalize", out);
    }
    return 0;
}

EqualizeHistOp::EqualizeHistOp(const json& param)
{
    ;//empty
}

int EqualizeHistOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out)
{
    cv::equalizeHist(image, out);
    if (_debug) {
        cc_print_mat("equalize_hist", out);
    }    
    return 0;
}

QuantizeOp::QuantizeOp(const json& param)
{
    json z = param.at("zero");
    for (json::iterator it = z.begin(); it != z.end(); ++it) {
        _zero.push_back(*it);
    }
    json s = param.at("step");
    for (json::iterator it = s.begin(); it != s.end(); ++it) {
        _step.push_back(*it);
    }
    CHECK (_zero.size() == _step.size());
    CHECK (_zero.size() == 1);
}

int QuantizeOp::run(cv::Mat& image, PreprocInfo& info, cv::Mat& out)
{
    int src_depth = image.type() & CV_MAT_DEPTH_MASK;
    if (src_depth != CV_32F) 
    {
        LOG(ERROR) << "Error: dtype not supported: " <<  src_depth;
        ABORT();
    }

    proc_opt::dlcv_image_to_int8(image, out, _step[0], _zero[0]);

    if (_debug) {
        cc_print_mat("quantize float into uint8", out);
    }    
    return 0;
}

// ToTensorOp 
ToTensorOp::ToTensorOp(const json& param){
    _hwc2chw     = param.at("hwc2chw");
    _swapchannel = param.at("swapchannel");
    _dtype       = param.at("dtype");
    if (!(_dtype == "float32" || _dtype == "int8")) {
        LOG(ERROR) << "Error: dtype not supported: " <<  _dtype;
        ABORT();
    }
}

inline void report_shape_mismatch(std::ostream &os, const cv::Mat& image, const Blob& blob) {
  os << "Blob size mismatch: image size: " << image.size() << " blob shape: ";
  for(int dim : blob.size()) {
      os << dim << ", ";
  }
}

/**
 * @brief copy data from cv::Mat to vision::Blob
 * 
 * @note Following copy are allowed:
 * - U8 Mat -> U8 Blob
 * - non-U8 Mat -> U8 Blob (calls image.convertTo(casted, CV_8U))
 * - F32 Mat -> F32 Blob
 * - non-F32 Mat -> F32 Blob (calls image.convertTo(casted, CV_32F))
 * if \p image and \p blob failed matching any routine, returns error.
 *  
 * @param image source image
 * @param blob target blob
 * @return int error code. \sa error_code.h
 */
inline int copy_mat_to_blob(const cv::Mat& image, Blob& blob) {
    uchar depth = image.type() & CV_MAT_DEPTH_MASK;
    if (blob.contains<uint8_t>()) {
        uint8_t* dest = blob.get<uint8_t>();
        if (depth == CV_8U) {
            // U8 Mat -> U8 Blob, just memcpy
            std::memcpy(dest, image.data, blob.byte_size());
        } else {
            // non-U8 Mat -> U8 Blob, may loss precision
            cv::Mat casted;
            image.convertTo(casted, CV_8U);
            std::memcpy(dest, casted.data, blob.byte_size());
        }
    } else if (blob.contains<float>()) {
        float* dest = blob.get<float>();
        if (depth == CV_32F) {
            // F32 Mat -> F32 Blob, just memcpy
            std::memcpy(dest, image.data, blob.byte_size());
        } else {
            // non-F32 Mat -> F32 Blob
            cv::Mat casted;
            image.convertTo(casted, CV_32F);
            std::memcpy(dest, casted.data, blob.byte_size());
        }
    } else {
        // check Inference::Inference then!
        return ERROR_CODE_IMAGE_DECODE;
    }
    return ERROR_CODE_SUCCESS;
}

/**
 * @brief check wheather \p image and \p blob are in N1HW or NHWC format
 * 
 * @param image source image
 * @param blob target \p vision::Blob
 * @return true if \p image and \p blob matches
 * @return false otherwise
 */
bool _is_n1hw_or_nhwc(const cv::Mat &image, const vision::Blob &blob) {
  const int H = image.rows;
  const int W = image.cols;
  const int C = image.channels();
  if (C != 1) { // is blob in {N, H, W, C}?
    return blob.size(1) == H and blob.size(2) == W and blob.size(3) == C;
  } else { // is blob in {N, 1, H, W} or {N, H, W, 1}
    return (blob.size(1) == 1 and blob.size(2) == H and blob.size(3) == W) 
        or (blob.size(1) == H and blob.size(2) == W and blob.size(3) == 1);
  }
}

int ToTensorOp::run(cv::Mat& image, PreprocInfo& info, vision::Blob &blob) {
    int H = image.rows;
    int W = image.cols;
    int C = image.channels();
    if (C == 1) {
        CHECK(_hwc2chw == false);
        CHECK(_swapchannel == false);
    }
    // if dtype is int8, image should be 8bit
    if (_dtype == "int8") {
        uchar depth = image.type() & CV_MAT_DEPTH_MASK;
        CHECK((depth == CV_8U and blob.contains<uint8_t>()) or
              (depth == CV_8S and blob.contains<int8_t>()));
    }
    if (_hwc2chw) {
        // ensure blob.size() is {N, C, H, W}
        if (blob.size(1) != C or blob.size(2) != H or blob.size(3) != W) {
            report_shape_mismatch(LOG(ERROR), image, blob);
            return ERROR_CODE_SHAPE_MISMATCH;
        }
        
        // HWC -> CHW for int8 & float32
        if (_dtype == "int8") {
            uint8_t *ptr = blob.get<uint8_t>();
            cv::Mat B(H, W, CV_8UC1, ptr);
            cv::Mat G(H, W, CV_8UC1, ptr + W * H); //TODO widthstep?
            cv::Mat R(H, W, CV_8UC1, ptr + W * H * 2);
            if (_swapchannel){ // swapchannel here for better performance
                std::vector<cv::Mat> data{R, G, B};
                cv::split(image, data);
            } else {
                std::vector<cv::Mat> data{B, G, R};
                cv::split(image, data);
            }
        } else { // float32
            float *ptr = blob.get<float>();
            cv::Mat B(H, W, CV_32FC1, ptr);
            cv::Mat G(H, W, CV_32FC1, ptr + W * H);
            cv::Mat R(H, W, CV_32FC1, ptr + W * H * 2);
            if (_swapchannel) { // swapchannel here for better performance
                std::vector<cv::Mat> data{R, G, B};
                cv::split(image, data);
            } else {
                std::vector<cv::Mat> data{B, G, R};
                cv::split(image, data);
            }
        }
    } else { // if (not _hwc2chw)
        // 1. ensure image is continuous. Mat::isContinuous checks image.flags internally.
        if (not image.isContinuous()) {
            LOG(ERROR) << "Image is not continuous";
            return ERROR_CODE_SHAPE_MISMATCH;
        }

        // 2. ensure blob.size() is {N, H, W, C} or {N, 1, H, W}
        if (not _is_n1hw_or_nhwc(image, blob)) {
            report_shape_mismatch(LOG(ERROR), image, blob);
            return ERROR_CODE_SHAPE_MISMATCH;
        }

        // 3. swap & copy
        int error_code = ERROR_CODE_SUCCESS;
        if (_swapchannel) {
            cv::Mat mat;
            cv::cvtColor(image, mat, cv::COLOR_RGB2BGR);
            error_code = copy_mat_to_blob(mat, blob);
        } else {
            error_code = copy_mat_to_blob(image, blob);
        }

        // 4. handle potential error
        if (error_code != ERROR_CODE_SUCCESS) {
            LOG(ERROR) << "Failed copy data from image to blob";
            return error_code;
        }
    }

    if (_debug) {
        if (blob.contains<uint8_t>()) {
            cc_print_blob<uint8_t>("ToTensor[uint8_t]", blob);
        }
        if (blob.contains<float>()) {
            cc_print_blob<float>("ToTensor[float32]", blob);
        }
    }
    return 0;
}




// Register all PreprocOp
typedef std::map<std::string, PreprocOp*(*)(const json& param)> PreprocMap;
PreprocMap preproc_map{
    {"convert_data_type",  &make_preproc<ConvertDataTypeOp> },
    {"resize",         &make_preproc<ResizeOp> },
    {"meanfile",       &make_preproc<MeanFileOp> },
    {"swapchannel",    &make_preproc<SwapChannelOp> },
    {"togray",         &make_preproc<ToGrayOp> },
    {"graytobgr",      &make_preproc<GrayToBGROp> },
    {"normalize",      &make_preproc<NormalizeOp> },
    {"batchnorm",      &make_preproc<BatchNormOp> },
    {"equalize_hist",  &make_preproc<EqualizeHistOp> },
    {"quantize",       &make_preproc<QuantizeOp> }
    // "totensor" is not here
};

// Preprocessor
Preprocessor::Preprocessor(const json& cfg){
    LOG(INFO) << "Preprocessor init";

    _debug = cfg.at("debug");
    vector<json> ps = cfg.at("ops");     
    int num = ps.size();

    // last op MUST be totensor
    string type = ps[num-1].at("type");
    json param  = ps[num-1].at("param");
    CHECK (type == "totensor");
    _to_tensor = new ToTensorOp(param);
    _to_tensor->_debug = _debug;

    for(int i = 0; i < num-1; i++){
        string type = ps[i].at("type");
        json param  = ps[i].at("param");
        //check valid op
        if (preproc_map.find(type) == preproc_map.end()){
            LOG(ERROR) << "Error: preproc op type not supported: " << type << ". Supported preproc ops are: ";
            for(PreprocMap::iterator it = preproc_map.begin(); it != preproc_map.end(); ++it) {
              LOG(ERROR) << "    " <<  it->first.c_str();
            }
            LOG(ERROR) << "    totensor";
            ABORT();
        }
        PreprocOp* op = preproc_map[type](param);
        op->_debug = _debug;
        _preproc_ops.push_back(op);
    } 
    _image_chain.resize(_preproc_ops.size());
}

int Preprocessor::run(cv::Mat& image, PreprocInfo& info, Blob &outdata){

    std::vector<cv::Mat*>  preproc_ptr(_image_chain.size() + 1);
    preproc_ptr[0]    = &image;
    for (unsigned int i = 0; i < _preproc_ops.size(); i++)
        preproc_ptr[i + 1] = &(_image_chain[i]);
    int       in_idx  = 0;
    int       out_idx = 0;
    for (unsigned int i = 0; i < _preproc_ops.size(); i++){
        int ret = _preproc_ops[i]->run(*(preproc_ptr[in_idx]), info, *(preproc_ptr[in_idx + 1]));
        if (ret != 0)
            return ret;
        out_idx ++;
        in_idx  ++;
    }
    int r = _to_tensor->run(*(preproc_ptr[out_idx]), info, outdata); //TODO only one network input supported
    if (r != 0)
        return r;

    return 0; 
}
Preprocessor::~Preprocessor(){
    if (_to_tensor){
        delete _to_tensor;
        _to_tensor = NULL;
    }
    for (unsigned int i =0; i < _preproc_ops.size(); i++){
        if (_preproc_ops[i] != NULL){
            delete _preproc_ops[i];
            _preproc_ops[i] = NULL;
        }
    }
}

}
