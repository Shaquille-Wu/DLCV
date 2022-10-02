/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file prior_box_layer.cpp
 * @brief the implementation for inference based on snpe
 * @author wuxiao@ainirobot.com
 * @date 2020-04-01
 */

#include "prior_box_layer.h"
#include <math.h>

namespace vision{

const float PriorBoxLayer::kAspectRatioDefault   = 1.0f;
const float PriorBoxLayer::kVarianceDefault      = 0.1f;
const float PriorBoxLayer::kFloatEpsilon         = 1e-6f;    

int PriorBoxLayer::Setup(const PriorBoxParam&   prior_box_param, 
                         int                    image_width, 
                         int                    image_height) noexcept
{
    int      i               = 0;
    int      j               = 0;
    float*   variance_start  = nullptr;
    int      layer_width     = (image_width / prior_box_param.step_size_) + 0.5f;
    int      layer_height    = (image_height / prior_box_param.step_size_) + 0.5f;
    if(((int)(layer_width * prior_box_param.step_size_ + 0.5f)) < image_width)
        layer_width = layer_width + 1;
    if(((int)(layer_height * prior_box_param.step_size_ + 0.5f)) < image_height)
        layer_height = layer_height + 1;

    if(layer_width <= 0 || layer_height <= 0 || image_width <= 0 || image_height <= 0 ||
        image_width < layer_width || image_height < layer_height)
    {
        return -1;
    }
        

    int      res             = validate_prior_box_param(prior_box_param);
    if(0 != res)    return res;

    prior_num_             = CalculatePriorNum(prior_box_param_);
    prior_box_total_count_ = CalculatePriorBoxCount(prior_box_param_, layer_width, layer_height);

    top_data_count_        = (prior_box_total_count_ * PriorBoxParam::kPriorBoxElementCount + prior_box_param.variance_.size() * prior_box_total_count_);
    if(nullptr != top_data_)
        delete top_data_;
    
    top_data_           = new float[top_data_count_];
    variance_start      = top_data_ + prior_box_total_count_ * PriorBoxParam::kPriorBoxElementCount;

    layer_width_        = layer_width;
    layer_height_       = layer_height;
    image_width_        = image_width;
    image_height_       = image_height;
    
    if(fabsf(prior_box_param_.step_size_) <= kFloatEpsilon)
    {
        step_x_         = static_cast<float>(image_width) / static_cast<float>(layer_width);
        step_y_         = static_cast<float>(image_height) / static_cast<float>(layer_height);
    }
    else
    {
        step_x_         = prior_box_param_.step_size_;
        step_y_         = prior_box_param_.step_size_;
    }


    GeneratePriorBox(prior_box_param_,           top_data_,      layer_width_, layer_height_, image_width_, image_height_, step_x_, step_y_);
    GenerateVariance(prior_box_param_.variance_, variance_start, layer_width_, layer_height_, prior_num_);

    return 0;
}

void PriorBoxLayer::Free() noexcept
{
    release();
}

int PriorBoxLayer::GeneratePriorBox(const PriorBoxParam&   prior_box_param, 
                                    float*                 prior_box_data,
                                    int                    layer_width,
                                    int                    layer_height,
                                    int                    image_width,
                                    int                    image_height,
                                    float                  step_x,
                                    float                  step_y) noexcept
{
    int    x = 0, y = 0, z = 0, w = 0;
    float  offset       = prior_box_param.offset_;
    int    data_ele_cnt = 0;
    for (y = 0; y < layer_height; y ++) 
    {
        for (x = 0; x < layer_width; x ++) 
        {
            float center_x = (x + offset) * step_x;
            float center_y = (y + offset) * step_y;
            float box_width = 0.0f, box_height = 0.0f;
            for (z = 0; z < (int)(prior_box_param.min_size_.size()); z ++) 
            {
                float min_size = prior_box_param.min_size_[z];
                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size;
                // xmin
                prior_box_data[data_ele_cnt++] = (center_x - 0.5f * box_width) / image_width;
                // ymin
                prior_box_data[data_ele_cnt++] = (center_y - 0.5f * box_height) / image_height;
                // xmax
                prior_box_data[data_ele_cnt++] = (center_x + 0.5f * box_width) / image_width;
                // ymax
                prior_box_data[data_ele_cnt++] = (center_y + 0.5f * box_height) / image_height;

                if (prior_box_param.max_size_.size() > 0) 
                {
                    float max_size = prior_box_param.max_size_[z];
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrtf(min_size * max_size);
                    // xmin
                    prior_box_data[data_ele_cnt++] = (center_x - 0.5f * box_width) / image_width;
                    // ymin
                    prior_box_data[data_ele_cnt++] = (center_y - 0.5f * box_height) / image_height;
                    // xmax
                    prior_box_data[data_ele_cnt++] = (center_x + 0.5f * box_width) / image_width;
                    // ymax
                    prior_box_data[data_ele_cnt++] = (center_y + 0.5f * box_height) / image_height;
                }

                // rest of priors
                for (w = 0; w < (int)(prior_box_param.aspect_ratio_.size()); w ++) 
                {
                    float ar = prior_box_param.aspect_ratio_[w];
                    if (fabs(ar - 1.0f) < kFloatEpsilon) 
                        continue;
                    box_width  = min_size * sqrtf(ar);
                    box_height = min_size / sqrtf(ar);
                    // xmin
                    prior_box_data[data_ele_cnt++] = (center_x - 0.5f * box_width) / image_width;
                    // ymin
                    prior_box_data[data_ele_cnt++] = (center_y - 0.5f * box_height) / image_height;
                    // xmax
                    prior_box_data[data_ele_cnt++] = (center_x + 0.5f * box_width) / image_width;
                    // ymax
                    prior_box_data[data_ele_cnt++] = (center_y + 0.5f * box_height) / image_height;
                }
            }
        }
    }

    // clip the prior's coordidate such that it is within [0, 1]
    if (true == prior_box_param.clip_) 
    {
        for (x = 0; x < data_ele_cnt; x++) 
        {
            if(prior_box_data[x] < 0.0f)
                prior_box_data[x] = 0.0f;
            else if(prior_box_data[x] > 1.0f)
                prior_box_data[x] = 1.0f;
        }
    }

    return (data_ele_cnt >> 2);
}

int PriorBoxLayer::GenerateVariance(const std::vector<float>&   variance, 
                                    float*                      variance_data,
                                    int                         layer_width,
                                    int                         layer_height,
                                    int                         prior_num) noexcept
{
    int   dim = layer_width * layer_height * prior_num * PriorBoxParam::kPriorBoxElementCount;
    int   i   = 0;

    if (variance.size() == 1) 
    {
        for(i = 0 ; i < dim ; i ++)
        {
            variance_data[i] = variance[0];
        }
    } 
    else 
    {
        int count = 0;
        for (int h = 0; h < layer_height; ++h) 
        {
            for (int w = 0; w < layer_width; ++w) 
            {
                for (int i = 0; i < prior_num; ++i) 
                {
                    for (int j = 0; j < 4; ++j) 
                    {
                        variance_data[count] = variance[j];
                        count++;
                    }
                }
            }
        }
    }
    return dim;
}

void PriorBoxLayer::release() noexcept
{
    layer_width_            = 0;
    layer_height_           = 0;
    image_width_            = 0;
    image_height_           = 0;
    step_x_                 = 0.0f;
    step_y_                 = 0.0f;        
    prior_num_              = 0;
    prior_box_total_count_  = 0;
    if(0 != top_data_)
        delete top_data_;
    top_data_         = nullptr;
    top_data_count_   = 0;
}

int PriorBoxLayer::validate_prior_box_param(const PriorBoxParam& other_prior_box_param)
{
    int   i             = 0 ;
    if(other_prior_box_param.min_size_.size() <= 0)  return -1;

    prior_box_param_.min_size_.clear();
    prior_box_param_.max_size_.clear();
    prior_box_param_.aspect_ratio_.clear();
    prior_box_param_.variance_.clear();
    for(i = 0 ; i < (int)(other_prior_box_param.min_size_.size()); i ++)
    {
        if(other_prior_box_param.min_size_[i] <= 0.0f)
        {
            return -1;
        }
        prior_box_param_.min_size_.push_back(other_prior_box_param.min_size_[i]);
    }

    prior_box_param_.aspect_ratio_.clear();
    prior_box_param_.aspect_ratio_.push_back(kAspectRatioDefault);
    prior_box_param_.flip_  = other_prior_box_param.flip_;

    for (i = 0; i < (int)(other_prior_box_param.aspect_ratio_.size()); i ++) 
    {
        float  cur_aspect_ratio = other_prior_box_param.aspect_ratio_[i];
        bool   already_exist    = false;
        for (int j = 0; j < (int)(prior_box_param_.aspect_ratio_.size()); j ++) 
        {
            if (fabsf(cur_aspect_ratio - prior_box_param_.aspect_ratio_[j]) < kFloatEpsilon) 
            {
                already_exist = true;
                break;
            }
        }
        if (false == already_exist) 
        {
            prior_box_param_.aspect_ratio_.push_back(cur_aspect_ratio);
            if (true == prior_box_param_.flip_) 
            {
                prior_box_param_.aspect_ratio_.push_back(1.0f/cur_aspect_ratio);
            }
        }
    }
    

    if (other_prior_box_param.max_size_.size() > 0) 
    {
        if(other_prior_box_param.min_size_.size() != other_prior_box_param.max_size_.size())
            return -1;
        for (i = 0; i < (int)(other_prior_box_param.max_size_.size()); i ++) 
        {
            prior_box_param_.max_size_.push_back(other_prior_box_param.max_size_[i]);
            if(other_prior_box_param.max_size_[i] <= other_prior_box_param.min_size_[i])
                return -1;
        }
    }
    prior_box_param_.clip_ = other_prior_box_param.clip_;
    if (other_prior_box_param.variance_.size() > 1) 
    {
        if(4 != other_prior_box_param.variance_.size())
            return -1;
        for (i = 0; i < (int)(other_prior_box_param.variance_.size()); ++i) 
        {
            if(other_prior_box_param.variance_[i] <= 0.0f)
                return -1;                
            prior_box_param_.variance_.push_back(other_prior_box_param.variance_[i]);
        }
    } 
    else if (other_prior_box_param.variance_.size() == 1) 
    {
        if(other_prior_box_param.variance_[0] <= 0.0f)
            return -1;
        prior_box_param_.variance_.push_back(other_prior_box_param.variance_[0]);
    } 
    else 
    {
        prior_box_param_.variance_.push_back(kVarianceDefault);
    }

    prior_box_param_.offset_ = other_prior_box_param.offset_;

    return 0;
};

}  //namespace vision