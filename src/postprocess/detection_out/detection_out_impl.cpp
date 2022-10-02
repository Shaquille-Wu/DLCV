/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detection_out_impl.cpp
 * @brief the implementation for detection_out
 * @author wuxiao@ainirobot.com
 * @date 2020-04-08
 */

#include "detection_out_impl.h"
#include "detection_out_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

namespace vision{

int DetectionOutImpl::Read(const char* orion_cfg_json) noexcept
{
    nlohmann::json  json;
    std::ifstream   input_stream(orion_cfg_json);
    if(false == input_stream.is_open())
        return DETECTION_OUT_ERROR_CODE::INVALID_JSON_FILE;
    input_stream >> json;

    return Read(json);
}

int DetectionOutImpl::Read(const nlohmann::json& param) noexcept
{
    reset_cfg_parameter();

    nlohmann::json new_param        = param;
    nlohmann::json json_image_info  = new_param["image_info"];
    if(false == json_image_info.is_null())
    {
        image_width_   = InputJson::SafeGet<int>(json_image_info["image_width"]);
        image_height_  = InputJson::SafeGet<int>(json_image_info["image_height"]);
    }

    vec_prior_box_param_               = InputJson::SafeGet<std::vector<PriorBoxParam> >(new_param["prior_box"]);
    detection_out_param_               = InputJson::SafeGet<DetectionOutParam>(new_param["detection_out"]);

    return DETECTION_OUT_ERROR_CODE::NONE;
}

int DetectionOutImpl::Setup() noexcept
{
    int      i                     = 0;
    int      res                   = 0;
    int      prior_box_layer_count = (int)(vec_prior_box_param_.size());
    prior_box_layer_.resize(prior_box_layer_count);
    for(i = 0 ; i < prior_box_layer_count ; i ++)
    {
        prior_box_layer_[i] = new PriorBoxLayer;

        res = prior_box_layer_[i]->Setup(vec_prior_box_param_[i], 
                                         image_width_, 
                                         image_height_);
        if(0 == res)
        {
            prior_box_data_count_ += prior_box_layer_[i]->GetTopDataCount();
            prior_box_count_      += prior_box_layer_[i]->GetPriorBoxCount();
        }
        else
        {
            ReleaseResource();
            return -1;
        }
    }

    detection_out_layer_  = new DetectionOutLayer;
    res = detection_out_layer_->Setup(prior_box_count_, detection_out_param_);
    if(0 != res)
    {
        ReleaseResource();
        return -1;
    }

    prior_box_data_ = new float[prior_box_data_count_];
    float*  cur_prior_box_data = prior_box_data_;
    float*  cur_variance_data  = prior_box_data_ + prior_box_count_ * PriorBoxParam::kPriorBoxElementCount;
    memset(prior_box_data_, 0, prior_box_data_count_ * sizeof(float));
    for(i = 0 ; i < prior_box_layer_count ; i ++)
    {
        int cur_prior_box_data_count = prior_box_layer_[i]->GetPriorBoxCount() * PriorBoxParam::kPriorBoxElementCount;
        int cur_prior_box_data_len   = cur_prior_box_data_count * sizeof(float);

        memcpy(cur_prior_box_data, 
                prior_box_layer_[i]->GetTopData(), 
                cur_prior_box_data_len);
        memcpy(cur_variance_data,  
                prior_box_layer_[i]->GetTopData() + cur_prior_box_data_count, 
                prior_box_layer_[i]->GetTopDataCount() * sizeof(float) - cur_prior_box_data_len);
        cur_prior_box_data  += cur_prior_box_data_count;
        cur_variance_data   += (prior_box_layer_[i]->GetTopDataCount() - cur_prior_box_data_count);
    }

    return 0;
}

int DetectionOutImpl::Solve(const float*   mbox_conf_data,
                            int            mbox_conf_data_count,
                            const float*   mbox_loc_data,
                            int            mbox_loc_data_count) noexcept
{
    float*  variance_data = prior_box_data_ + prior_box_count_ * PriorBoxParam::kPriorBoxElementCount;
    int     res           = detection_out_layer_->Solve(prior_box_data_, 
                                                        prior_box_count_, 
                                                        variance_data, 
                                                        4, 
                                                        mbox_conf_data, 
                                                        mbox_conf_data_count, 
                                                        mbox_loc_data, 
                                                        mbox_loc_data_count);

    return res;
}

void DetectionOutImpl::ReleaseResource() noexcept
{
    int i = 0 ;

    for(i = 0 ; i < (int)(prior_box_layer_.size()); i ++)
    {
        if(nullptr != prior_box_layer_[i])
        {
            prior_box_layer_[i]->Free();
            delete prior_box_layer_[i];
        }
    }
    prior_box_layer_.clear();

    if(nullptr != detection_out_layer_)
    {
        detection_out_layer_->Free();
        delete detection_out_layer_;
    }
    detection_out_layer_ = nullptr;

    if(nullptr != prior_box_data_)
        delete prior_box_data_;
    prior_box_data_       = nullptr;
    prior_box_count_      = 0;
    prior_box_data_count_ = 0;    
}

void DetectionOutImpl::reset_cfg_parameter()
{
    image_width_       = 0;
    image_height_      = 0;    
    vec_prior_box_param_.clear();
    conf_tensor_       = "";
    loc_tensor_        = "";
}

} //namespace vision