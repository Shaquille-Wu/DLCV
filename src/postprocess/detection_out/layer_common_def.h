#ifndef  __ORION_SNPE_LAYER_DEF_H__
#define  __ORION_SNPE_LAYER_DEF_H__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file orion_snpe_layer_common_def.h
 * @brief This header file defines common data type
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-04-01
 */

#include <vector>
#include <json.hpp>
#include "safe_json.h"

namespace vision{

/**
 * @brief PriorBoxParam 
 * 
 */
class PriorBoxParam
{
public:
    static const  int                 kPriorBoxElementCount = 4;

public:
    /**
     * @brief default constructor for PriorBoxParam
     * 
     * @exceptsafe No throw.
     */
    PriorBoxParam() noexcept
    {
        min_size_.clear();
        max_size_.clear();
        aspect_ratio_.clear();        
        flip_       = true;
        clip_       = false;
        variance_.clear();
        step_size_  = 0.0f;
        offset_     = 0.5f;
        layer_size_ = 0;
    };

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */  
    virtual ~PriorBoxParam() noexcept
    {
        min_size_.clear();
        max_size_.clear();
        aspect_ratio_.clear();
        flip_       = true;
        clip_       = false;
        variance_.clear();
        step_size_  = 0.0f;
        offset_     = 0.5f;
        layer_size_ = 0;
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other PriorBoxParam object.
     * 
     * @exceptsafe No throw.
     */
    PriorBoxParam(const PriorBoxParam& other) noexcept
    {
        min_size_      = other.min_size_;
        max_size_      = other.max_size_;
        aspect_ratio_  = other.aspect_ratio_;
        flip_          = other.flip_;
        clip_          = other.clip_;
        variance_      = other.variance_;
        step_size_     = other.step_size_;
        offset_        = other.offset_;
        layer_size_    = other.layer_size_;
    };

    /**
     * @brief Construct from parameter
     *
     * @param buffer A BasicBuffer object to reference.
     * 
     * @param min_size array of min size
     * 
     * @param max_size array of max size
     * 
     * @param aspect_ratio array of aspect ratio
     * 
     * @param flip flag for aspect_ratio <--> 1/aspect_ratio
     * 
     * @param clip flag for clip bouding box
     * 
     * @param variance array of variance
     * 
     * @param offset offset for center, default is 0.5
     * 
     * @param layer_size layer size
     * 
     * @exceptsafe No throw.
     */
    PriorBoxParam(const std::vector<float>&   min_size,
                  const std::vector<float>&   max_size,
                  const std::vector<float>&   aspect_ratio,
                  bool                        flip,
                  bool                        clip,
                  const std::vector<float>&   variance,
                  float                       step_size,
                  float                       offset,
                  int                         layer_size) noexcept
    {
        min_size_      = min_size;
        max_size_      = max_size;
        aspect_ratio_  = aspect_ratio;
        flip_          = flip;
        clip_          = clip;
        variance_      = variance;
        step_size_     = step_size;
        offset_        = offset;
        layer_size_    = layer_size;
    };

    /**
     * @brief assignment constructor
     * 
     * @param other The other PriorBoxParam object.
     * 
     * @exceptsafe No throw.
     */
    PriorBoxParam& operator=(const PriorBoxParam&  other)
    {
        if(this == &other)
            return *this;

        min_size_      = other.min_size_;
        max_size_      = other.max_size_;
        aspect_ratio_  = other.aspect_ratio_;
        flip_          = other.flip_;
        clip_          = other.clip_;
        variance_      = other.variance_;
        step_size_     = other.step_size_;
        offset_        = other.offset_;
        layer_size_    = other.layer_size_;

        return *this;
    };

public:
    /**
     * @brief min size array
     */    
    std::vector<float>            min_size_;

    /**
     * @brief max size array, if max_size_.size() is not zero, max_size_.size() should be equal to min_size_.size()
     */     
    std::vector<float>            max_size_ ;

    /**
     * @brief aspect ratio array
     */  
    std::vector<float>            aspect_ratio_;

    /**
     * @brief flag for aspect_ratio <--> 1/aspect_ratio
     */     
    bool                          flip_;

    /**
     * @brief flag for clip bouding box
     */      
    bool                          clip_;

    /**
     * @brief variance array
     */      
    std::vector<float>            variance_;

     /**
     * @brief step size, it should be compute through image_size/layer_size, if it is 0.0f
     */    
    float                         step_size_;

    /**
     * @brief offset for center, default is 0.5
     */     
    float                         offset_;

    /**
     * @brief layer size, there are some relationship between step_size_ and layer_size
     */     
    int                           layer_size_;    
}; //PriorBoxParam

inline void from_json(const nlohmann::json& j, PriorBoxParam& prior_box_param) noexcept
{
    prior_box_param.min_size_      = InputJson::SafeGet<std::vector<float> >(j["min_size"]);
    prior_box_param.max_size_      = InputJson::SafeGet<std::vector<float> >(j["max_size"]);
    prior_box_param.aspect_ratio_  = InputJson::SafeGet<std::vector<float> >(j["aspect_ratio"]);
    prior_box_param.flip_          = InputJson::SafeGet<bool>(j["flip"]);
    prior_box_param.clip_          = InputJson::SafeGet<bool>(j["clip"]);
    prior_box_param.variance_      = InputJson::SafeGet<std::vector<float> >(j["variance"]);
    prior_box_param.step_size_     = InputJson::SafeGet<float>(j["step_size"]);
    prior_box_param.offset_        = InputJson::SafeGet<float>(j["offset"]);
    prior_box_param.layer_size_    = InputJson::SafeGet<int>(j["layer_size"]);
};

typedef enum tag_priorbox_code_type{
    CODE_TYPE_CORNER = 0,
    CODE_TYPE_CENTER_SIZE,
    CODE_TYPE_CORNER_SIZE,
    CODE_TYPE_SUM
}PRIOR_BOX_CODE_TYPE;

static const char*  kPriorBoxCodeTypeStr[CODE_TYPE_SUM] = {
    "CORNER",
    "CENTER_SIZE",
    "CORNER_SIZE",
};

class DetectionOutParam{

public:
    /**
     * @brief default constructor for DetectionOutParam
     * 
     * @exceptsafe No throw.
     */
    DetectionOutParam() noexcept
    {
        class_count_          = 1 ;
        background_label_id_  = 0 ;
        conf_threshold_       = 0.0f;
        nms_threshold_        = 0.0f;
        top_k_                = 0 ;
        bbox_code_type_       = CODE_TYPE_CENTER_SIZE ;
    };

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */  
    virtual ~DetectionOutParam() noexcept
    {
        class_count_          = 1 ;
        background_label_id_  = 0 ;
        conf_threshold_       = 0.0f;
        nms_threshold_        = 0.0f;
        top_k_                = 0 ;
        bbox_code_type_       = CODE_TYPE_CENTER_SIZE ;
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other DetectionOutParam object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutParam(const DetectionOutParam& other) noexcept
    {
        class_count_          = other.class_count_ ;
        background_label_id_  = other.background_label_id_ ;
        conf_threshold_       = other.conf_threshold_ ;
        nms_threshold_        = other.nms_threshold_ ;
        top_k_                = other.top_k_ ;
        bbox_code_type_       = other.bbox_code_type_ ;
    };

    /**
     * @brief Construct from parameter
     *
     * @param class_count class count for result.
     * 
     * @param background_label_id background label id
     * 
     * @param conf_threshold confidence threshold for result
     * 
     * @param nms_threshold nms threshold for result
     * 
     * @param top_k select best k results
     * 
     * @param bbox_code_type bbox decode type
     * 
     * @exceptsafe No throw.
     */
    DetectionOutParam(int        class_count,
                      int        background_label_id,
                      float      conf_threshold,
                      float      nms_threshold,
                      int        top_k,
                      int        bbox_code_type) noexcept
    {
        class_count_          = class_count ;
        background_label_id_  = background_label_id ;
        conf_threshold_       = conf_threshold ;
        nms_threshold_        = nms_threshold ;
        top_k_                = top_k ;
        bbox_code_type_       = bbox_code_type ;
    };

    /**
     * @brief assignment constructor
     * 
     * @param other The other DetectionOutParam object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutParam& operator=(const DetectionOutParam&  other)
    {
        if(this == &other)
            return *this;

        class_count_          = other.class_count_ ;
        background_label_id_  = other.background_label_id_ ;
        conf_threshold_       = other.conf_threshold_ ;
        nms_threshold_        = other.nms_threshold_ ;
        top_k_                = other.top_k_ ;
        bbox_code_type_       = other.bbox_code_type_ ;

        return *this;    
    };

public:
    /**
     * @brief class count in result
     */ 
    int                          class_count_;

    /**
     * @brief background label id
     */ 
    int                          background_label_id_;

    /**
     * @brief confidence threshold for result
     */    
    float                        conf_threshold_;

    /**
     * @brief nms threshold for result
     */      
    float                        nms_threshold_;

    /**
     * @brief select best k results
     */ 
    int                          top_k_;

     /**
     * @brief bbox decode type
     */    
    int                          bbox_code_type_;
}; //DetectionOutParam

inline void from_json(const nlohmann::json& j, DetectionOutParam& detection_out_param) noexcept
{
    detection_out_param.class_count_          = InputJson::SafeGet<int>(j["class_count"]);
    detection_out_param.background_label_id_  = InputJson::SafeGet<int>(j["background_label_id"]);
    detection_out_param.conf_threshold_       = InputJson::SafeGet<float>(j["conf_threshold"]);
    detection_out_param.nms_threshold_        = InputJson::SafeGet<float>(j["nms_threshold"]);
    detection_out_param.top_k_                = InputJson::SafeGet<int>(j["top_k"]);
    std::string  str_code_type                = InputJson::SafeGet<std::string>(j["bbox_code_type"]);
    int          i                            = 0;
    for(i = 0 ; i < PRIOR_BOX_CODE_TYPE::CODE_TYPE_SUM ; i ++)
    {
        if(str_code_type == kPriorBoxCodeTypeStr[i])
            break;
    }
    if(i >= PRIOR_BOX_CODE_TYPE::CODE_TYPE_SUM)
        i = 0;
    detection_out_param.bbox_code_type_       = i;
};

class QuantizationParam
{
public:
    /**
     * @brief default constructor for QuantizationParam
     * 
     * @exceptsafe No throw.
     */
    QuantizationParam() noexcept
    {
        zero_ = 0;
        step_ = 0.0f;
    };

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */  
    virtual ~QuantizationParam() noexcept
    {
        zero_ = 0;
        step_ = 0.0f;
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other QuantizationParam object.
     * 
     * @exceptsafe No throw.
     */
    QuantizationParam(const QuantizationParam& other) noexcept
    {
        zero_  = other.zero_;
        step_  = other.step_;
    };

    /**
     * @brief Construct from parameter
     *
     * @param zero value of quantize zero
     * 
     * @param step value of quantize step size
     * 
     * @exceptsafe No throw.
     */
    QuantizationParam(int zero, float step) noexcept
    {
        zero_  = zero;
        step_  = step;
    };

    /**
     * @brief assignment constructor
     * 
     * @param other The other QuantizationParam object.
     * 
     * @exceptsafe No throw.
     */
    QuantizationParam& operator=(const QuantizationParam&  other)
    {
        if(this == &other)
            return *this;

        zero_  = other.zero_;
        step_  = other.step_;

        return *this;    
    };

public:
    /**
     * @brief value of quantize zero
     */ 
    int                          zero_;

     /**
     * @brief value of quantize step
     */    
    float                        step_;

}; //QuantizationParam

inline void from_json(const nlohmann::json& j, QuantizationParam& quantization_param) noexcept
{
    quantization_param.zero_  = InputJson::SafeGet<int>(j["zero"]);
    quantization_param.step_  = InputJson::SafeGet<float>(j["step"]);
};

} //namespace vision

#endif