#ifndef  __ORION_DETECTION_OUT_IMPL_H__
#define  __ORION_DETECTION_OUT_IMPL_H__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detection_out_impl.h
 * @brief This header file defines detection_out's implementation
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-04-08
 */

#include "layer_common_def.h"
#include "./prior_box_layer.h"
#include "./detection_out_layer.h"
#include <string.h>
#include <string>


namespace vision{

/**
 * @brief DetectionOutImpl 
 * 
 */
class DetectionOutImpl
{
public:
    /**
     * @brief default constructor for DetectionOutImpl
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl() noexcept
    {
        image_width_           = 0;
        image_height_          = 0;

        detection_out_layer_   = nullptr;
        prior_box_data_        = nullptr;
        prior_box_count_       = 0;
        prior_box_data_count_  = 0;
    };

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */  
    virtual ~DetectionOutImpl() noexcept
    { 
        ReleaseResource();
        reset_cfg_parameter();
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other DetectionOutImpl object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl(const DetectionOutImpl& other) noexcept
    {
        image_width_           = other.image_width_;
        image_height_          = other.image_height_;
        vec_prior_box_param_   = other.vec_prior_box_param_;
        detection_out_param_   = other.detection_out_param_;
        quantization_param_    = other.quantization_param_;   
        conf_tensor_           = other.conf_tensor_;
        loc_tensor_            = other.loc_tensor_;
        prior_box_layer_.resize(other.prior_box_layer_.size());
        for(int i = 0 ; i < (int)(other.prior_box_layer_.size()) ; i ++)
            prior_box_layer_[i] = new PriorBoxLayer(*(other.prior_box_layer_[i]));

        detection_out_layer_   = 0;
        if(nullptr != other.detection_out_layer_)
            detection_out_layer_ = new DetectionOutLayer(*(other.detection_out_layer_));
        prior_box_data_        = nullptr;
        if(nullptr != other.prior_box_data_)
        {
            prior_box_data_ = new float[other.prior_box_data_count_];
            memcpy(prior_box_data_, other.prior_box_data_, other.prior_box_data_count_ * sizeof(float));
        }
        prior_box_count_       = other.prior_box_count_;
        prior_box_data_count_  = other.prior_box_data_count_;          
    };

    /**
     * @brief Construct from parameter
     *
     * @param image_width image width for model.
     * 
     * @param image_height image height for model.
     * 
     * @param vec_prior_box_param array of prior_box's parameter
     * 
     * @param detection_out_param detection_out's parameter
     * 
     * @param quantization_param parameter for quantization
     * 
     * @param conf_tensor confidence tensor
     * 
     * @param loc_tensor location tensor
     * 
     * @param result_tensor result tensor
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl(int                                 image_width,
                     int                                 image_height,
                     const std::vector<PriorBoxParam>&   vec_prior_box_param,
                     const DetectionOutParam&            detection_out_param,
                     const QuantizationParam&            quantization_param,
                     const std::string&                  conf_tensor,
                     const std::string&                  loc_tensor) noexcept
    {
        image_width_           = image_width;
        image_height_          = image_height;
        vec_prior_box_param_   = vec_prior_box_param;
        detection_out_param_   = detection_out_param;
        quantization_param_    = quantization_param;
        conf_tensor_           = conf_tensor;
        loc_tensor_            = loc_tensor;

        detection_out_layer_   = nullptr;
        prior_box_data_        = nullptr;
        prior_box_count_       = 0;
        prior_box_data_count_  = 0;          
    };

    /**
     * @brief assignment constructor
     * 
     * @param other The other DetectionOutImpl object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl& operator=(const DetectionOutImpl&  other) noexcept
    {
        if(this == &other)
            return *this;

        image_width_         = other.image_width_;
        image_height_        = other.image_height_;
        vec_prior_box_param_ = other.vec_prior_box_param_;
        detection_out_param_ = other.detection_out_param_;
        quantization_param_  = other.quantization_param_;
        conf_tensor_         = other.conf_tensor_;
        loc_tensor_          = other.loc_tensor_; 
        for(int i = 0 ; i < (int)(prior_box_layer_.size()) ; i ++)
        {
            if(nullptr != prior_box_layer_[i])
                delete prior_box_layer_[i];
        }
        prior_box_layer_.resize(other.prior_box_layer_.size());
        for(int i = 0 ; i < (int)(other.prior_box_layer_.size()) ; i ++)
            prior_box_layer_[i] = new PriorBoxLayer(*(other.prior_box_layer_[i]));

        detection_out_layer_   = 0;
        if(nullptr != other.detection_out_layer_)
            detection_out_layer_ = new DetectionOutLayer(*(other.detection_out_layer_));
        prior_box_data_        = nullptr;
        if(nullptr != other.prior_box_data_)
        {
            prior_box_data_ = new float[other.prior_box_data_count_];
            memcpy(prior_box_data_, other.prior_box_data_, other.prior_box_data_count_ * sizeof(float));
        }
        prior_box_count_       = other.prior_box_count_;
        prior_box_data_count_  = other.prior_box_data_count_; 

        return *this;    
    }; 

    /**
     * @brief read content from json configuration file(.json)
     *
     * @param orion_cfg_json file name of orion configuration file
     * 
     * @return status of read, 0 is ok, other is failed.
     * 
     * @exceptsafe No throw.
     */
    int                                  Read(const char* orion_cfg_json) noexcept;

    /**
     * @brief read content from json object
     *
     * @param param json object for configuration file
     * 
     * @return status of read, 0 is ok, other is failed.
     * 
     * @exceptsafe No throw.
     */
    int                                  Read(const nlohmann::json& param) noexcept;

    /**
     * @brief setup post-process according initialized parameter.
     * 
     * @return setup's status, 0 is ok, other is failed
     * 
     * @exceptsafe No throw.
     * 
     * @note the initial parameter including: image_width_, image_height_, 
     *                                        vec_prior_box_param_, 
     *                                        detection_out_param_, 
     *                                        quantization_param_, 
     *                                        conf_tensor_, loc_tensor_, result_tensor_
     *       there are 2 methods for the initialization of above parameters:
     *       1).Read(json);
     *       2).calling SetXXXX member function;
     */
    int                                  Setup() noexcept;


    /**
     * @brief solve result.
     * 
     * @param mbox_conf_data mbox_conf's data, the size of its buffer should be: mbox_conf_data_count * sizeof(float).
     * 
     * @param mbox_conf_data_count count of mbox_conf_data
     * 
     * @param mbox_loc_data mbox_loc's data, the size of its buffer should be: mbox_loc_data_count * sizeof(float).
     * 
     * @param mbox_loc_data_count count of mbox_loc_data
     * 
     * @return setup's status, 0 is ok, other is failed
     *
     * @exceptsafe No throw.
     */
    int                                  Solve(const float*   mbox_conf_data,
                                               int            mbox_conf_data_count,
                                               const float*   mbox_loc_data,
                                               int            mbox_loc_data_count) noexcept;

    /**
     * @brief get result data, including priorbox and its variance
     *
     * @return layer's top data
     *
     * @exceptsafe No throw.
     */
    const float*                         GetResultData() const noexcept         { return detection_out_layer_->GetTopData(); };   

    /**
     * @brief get count of result
     *
     * @return count of result
     *
     * @exceptsafe No throw.
     */   
    int                                  GetResultItemCount() const noexcept    { return detection_out_layer_->GetResultItemCount(); } ;

    /**
     * @brief get count of result data, 
     *
     * @return count of data buf
     *
     * @exceptsafe No throw.
     */       
    int                                  GetResultDataCount() const noexcept    { return detection_out_layer_->GetResultItemCount() * DetectionOutLayer::kResultItemElementCount; } ;                                    

    /**
     * @brief release layers' resource 
     *
     * @return none
     *
     * @exceptsafe No throw.
     */  
    void                                 ReleaseResource() noexcept;

    /**
     * @brief get count of prior boxs, the value is valid after Setup
     *
     * @return count of prior boxs
     *
     * @exceptsafe No throw.
     */  
    int                                  GetPriorBoxCount() const noexcept      { return prior_box_count_; };

    /**
     * @brief get image width
     *
     * @return image width for model.
     * 
     * @exceptsafe No throw.
     */
    int                                  GetImageWidth() const noexcept         { return image_width_; };

    /**
     * @brief set image width, it should be executed befor Setup
     *
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                    SetImageWidth(int image_width) noexcept      
    { 
        image_width_ = image_width;
        return *this;
    }; 

    /**
     * @brief get image height.
     *
     * @return image height for model.
     * 
     * @exceptsafe No throw.
     */
    int                                  GetImageHeight() const noexcept        { return image_height_; };

    /**
     * @brief set image height, it should be executed befor Setup
     *
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                    SetImageHeight(int image_height) noexcept      
    { 
        image_height_ = image_height;
        return *this;
    }; 

    /**
     * @brief get all of prior_boxs's parameter.
     *
     * @return all of prior_boxs's parameter.
     * 
     * @exceptsafe No throw.
     */
    const std::vector<PriorBoxParam>&    GetPriorBoxParam() const noexcept      { return vec_prior_box_param_; };

    /**
     * @brief set all of prior_boxs's parameter, it should be executed befor Setup
     *
     * @param prior_box_param prior box paramter.
     * 
     * @return reference of itself.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                    SetPriorBoxParam(const std::vector<PriorBoxParam>& prior_box_param) noexcept      
    { 
        vec_prior_box_param_ = prior_box_param;
        return *this;
    };    

    /**
     * @brief get detection_out's parameter.
     *
     * @return detection_out's parameter.
     * 
     * @exceptsafe No throw.
     */
    const DetectionOutParam&              GetDetectionOutParam() const noexcept  { return detection_out_param_; };

    /**
     * @brief set all of prior_boxs's parameter, it should be executed befor Setup
     *
     * @param detection_out_param parameter for detection_out
     * 
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                     SetDetectionOutParam(const DetectionOutParam& detection_out_param) noexcept      
    { 
        detection_out_param_ = detection_out_param;
        return *this;
    };

    /**
     * @brief get parameter for quantized, including zero and step.
     *
     * @return reference of quantization parameter.
     * 
     * @exceptsafe No throw.
     */  
    const QuantizationParam&              GetQuantizationParameter() const noexcept      { return quantization_param_; };

    /**
     * @brief set quantize step, it should be executed befor Setup
     *
     * @param quantization_param parameter for quantization
     * 
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                     SetQuantizationParameter(const QuantizationParam& quantization_param) noexcept      
    { 
        quantization_param_ = quantization_param;
        return *this;
    };

    /**
     * @brief get confidence tensor name
     *
     * @return confidence tensor name
     * 
     * @exceptsafe No throw.
     */
    const std::string&                    GetOutputConfTensor() const noexcept    { return conf_tensor_; };

    /**
     * @brief set confidence tensor name, it should be executed befor Setup
     *
     * @param conf_tensor name of confidence tensor
     * 
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                     SetOutputConfTensor(const std::string& conf_tensor) noexcept      
    { 
        conf_tensor_ = conf_tensor;
        return *this;
    };

    /**
     * @brief get location tensor name
     *
     * @return location tensor name
     * 
     * @exceptsafe No throw.
     */
    const std::string&                    GetOutputLocTensor() const noexcept    { return loc_tensor_; };

    /**
     * @brief set location tensor name, it should be executed befor Setup
     *
     * @param loc_tensor name of location tensor
     * 
     * @return reference of itself
     * 
     * @exceptsafe No throw.
     */
    DetectionOutImpl&                     SetOutputLocTensor(const std::string& loc_tensor) noexcept      
    { 
        loc_tensor_ = loc_tensor;
        return *this;
    };

private:
    void                                  reset_cfg_parameter();


protected:
    int                                   image_width_;
    int                                   image_height_;
    std::vector<PriorBoxParam>            vec_prior_box_param_;
    DetectionOutParam                     detection_out_param_;
    QuantizationParam                     quantization_param_;
    std::string                           conf_tensor_;
    std::string                           loc_tensor_;

    std::vector<PriorBoxLayer*>           prior_box_layer_;
    DetectionOutLayer*                    detection_out_layer_;
    float*                                prior_box_data_;
    int                                   prior_box_count_;
    int                                   prior_box_data_count_;    
}; //PostProcess

} //namespace vision


#endif