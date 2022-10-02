#ifndef  __PRIOR_BOX_LAYER__
#define  __PRIOR_BOX_LAYER__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file prior_box_layer.h
 * @brief This header file defines compact prior box layer
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-04-01
 */

#include "layer_common_def.h"
#include <string.h>

namespace vision{

/**
 * @brief PriorBoxLayer 
 * 
 */
class PriorBoxLayer
{
public:
    static const  float    kAspectRatioDefault;
    static const  float    kVarianceDefault;
    static const  float    kFloatEpsilon;

public:
    /**
     * @brief default constructor for PriorBoxLayer
     * 
     * @exceptsafe No throw.
     */
    PriorBoxLayer() noexcept
    {
        layer_width_            = 0;
        layer_height_           = 0;
        image_width_            = 0;
        image_height_           = 0;
        step_x_                 = 0.0f;
        step_y_                 = 0.0f;        
        prior_num_              = 0;
        prior_box_total_count_  = 0;
        top_data_               = 0;
        top_data_count_         = 0;
    };

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */  
    virtual ~PriorBoxLayer() noexcept
    {
        release();
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other PriorBoxLayer object.
     * 
     * @exceptsafe No throw.
     */
    PriorBoxLayer(const PriorBoxLayer& other) noexcept
    {
        prior_box_param_        = other.prior_box_param_;
        layer_width_            = other.layer_width_;
        layer_height_           = other.layer_height_;
        image_width_            = other.image_width_;
        image_height_           = other.image_height_;
        step_x_                 = other.step_x_;
        step_y_                 = other.step_y_;
        prior_num_              = other.prior_num_;
        prior_box_total_count_  = other.prior_box_total_count_;
        if(nullptr != top_data_)
            delete top_data_;       
        top_data_        = nullptr;
        top_data_count_  = 0;
        if(nullptr != other.top_data_)
        {
            top_data_         = new float[other.top_data_count_];
            top_data_count_   = other.top_data_count_;
            memcpy(top_data_, other.top_data_, top_data_count_ * sizeof(float));
        }
    };

    /**
     * @brief assignment constructor
     * 
     * @param other The other PriorBoxLayer object.
     * 
     * @exceptsafe No throw.
     */
    PriorBoxLayer& operator=(const PriorBoxLayer&  other) noexcept
    {
        if(&other == this)
            return *this;
        
        prior_box_param_        = other.prior_box_param_;
        layer_width_            = other.layer_width_;
        layer_height_           = other.layer_height_;
        image_width_            = other.image_width_;
        image_height_           = other.image_height_;
        step_x_                 = other.step_x_;
        step_y_                 = other.step_y_;        
        prior_box_total_count_  = other.prior_box_total_count_;

        if(nullptr != top_data_)
            delete top_data_;
        top_data_        = nullptr;
        top_data_count_  = 0;            
        if(nullptr != other.top_data_)
        {
            top_data_       = new float[other.top_data_count_];
            top_data_count_ = other.top_data_count_;
            memcpy(top_data_, other.top_data_, top_data_count_ * sizeof(float));
        }
        return *this;
    };


    /**
     * @brief setup prior box according to these parameters.
     *
     * @param prior_box_param prior box param.
     * 
     * @param image_width source image's width
     * 
     * @param image_height source image's height
     * 
     * @return setup's status, 0 is ok, other is failed
     *
     * @exceptsafe No throw.
     */
    int                     Setup(const PriorBoxParam&   prior_box_param, 
                                  int                    image_width, 
                                  int                    image_height) noexcept;

    /**
     * @brief release all of resource in prior_box_layer
     * 
     * @return none
     *
     * @exceptsafe No throw.
     */    
    void                    Free() noexcept;

    /**
     * @brief Get prior_box_param_.
     *
     * @return prior box parameter.
     *
     * @exceptsafe No throw.
     */
    const PriorBoxParam&    GetPriorBoxParam() const noexcept     { return prior_box_param_; };

    /**
     * @brief Get layer's width.
     *
     * @return layer's width.
     *
     * @exceptsafe No throw.
     */
    int                     GetLayerWidth() const noexcept        { return layer_width_; };

    /**
     * @brief Get layer's height.
     *
     * @return layer's height.
     *
     * @exceptsafe No throw.
     */
    int                     GetLayerHeight() const noexcept       { return layer_height_; };

    /**
     * @brief Get width of source image
     *
     * @return width of source image
     *
     * @exceptsafe No throw.
     */
    int                     GetImageWidth() const noexcept        { return image_width_; };

    /**
     * @brief Get height of source image
     *
     * @return height of source image
     *
     * @exceptsafe No throw.
     */
    int                     GetImageHeight() const noexcept       { return image_height_; };

    /**
     * @brief Get count of prior box
     *
     * @return count of prior box
     *
     * @exceptsafe No throw.
     */
    int                     GetPriorBoxCount() const noexcept     { return prior_box_total_count_; };   

    /**
     * @brief Get number of prior
     *
     * @return number of prior
     *
     * @exceptsafe No throw.
     */
    int                     GetPriorNum() const noexcept          { return prior_num_; };

    /**
     * @brief calculate the prior number
     *
     * @param prior_box_param parameter of prior box
     * 
     * @return count of prior box
     *
     * @exceptsafe No throw.
     */
    static int              CalculatePriorNum(const PriorBoxParam& prior_box_param) noexcept
    {
        int   prior_box_cnt  = 0;
        prior_box_cnt        = prior_box_param.aspect_ratio_.size() * prior_box_param.min_size_.size();
        prior_box_cnt       += prior_box_param.max_size_.size();

        return prior_box_cnt;
    };

    /**
     * @brief calculate the count of prior box which generated by this layer
     *
     * @param prior_box_param parameter of prior box
     * 
     * @param layer_width layer's width
     * 
     * @param layer_height layer's height
     * 
     * @return count of prior box
     *
     * @exceptsafe No throw.
     */
    static int              CalculatePriorBoxCount(const PriorBoxParam& prior_box_param, int layer_width, int layer_height) noexcept
    {
        return layer_height * layer_width * CalculatePriorNum(prior_box_param);
    };

    /**
     * @brief calculate the count of prior box which generated by this layer
     *
     * @param prior_box_param parameter of prior box
     * 
     * @param layer_width layer's width
     * 
     * @param layer_height layer's height
     * 
     * @param image_width source image's width
     * 
     * @param image_height source image's height
     * 
     * @param step_x image_width/layer_width
     * 
     * @param step_y image_height/layer_height
     * 
     * @return count of prior box
     *
     * @exceptsafe No throw.
     */
    static int              GeneratePriorBox(const PriorBoxParam&   prior_box_param, 
                                             float*                 prior_box_data,
                                             int                    layer_width,
                                             int                    layer_height,
                                             int                    image_width,
                                             int                    image_height,
                                             float                  step_x,
                                             float                  step_y) noexcept;

    /**
     * @brief calculate the count of prior box which generated by this layer
     *
     * @param prior_box_param parameter of prior box
     * 
     * @param layer_width layer's width
     * 
     * @param layer_height layer's height
     * 
     * @param prior_num number of priors
     * 
     * @return count of variance's data
     *
     * @exceptsafe No throw.
     */
    static int              GenerateVariance(const std::vector<float>&   variance, 
                                             float*                      variance_data,
                                             int                         layer_width,
                                             int                         layer_height,
                                             int                         prior_num) noexcept;

    /**
     * @brief get layer's top data, including priorbox and its variance
     *
     * @return layer's top data
     *
     * @exceptsafe No throw.
     */
    const float*            GetTopData() const noexcept                            { return top_data_; };

    /**
     * @brief get count of top data, including priorbox and its variance, it should be as following:
     *        top_data_count_ = (prior_box_total_count_ * kPriorBoxElementCount + (prior_box_total_count_ * prior_box_param_.variance_.size()))
     *
     * @return count of data buf
     *
     * @exceptsafe No throw.
     */   
    int                     GetTopDataCount() const noexcept                      { return top_data_count_; };

private:
    void                    release() noexcept;

    int                     validate_prior_box_param(const PriorBoxParam& other_prior_box_param);

private:
    PriorBoxParam           prior_box_param_;
    int                     layer_width_;
    int                     layer_height_;
    int                     image_width_;
    int                     image_height_;
    float                   step_x_;
    float                   step_y_;
    int                     prior_num_;
    int                     prior_box_total_count_;
    float*                  top_data_;
    int                     top_data_count_;
}; //PriorBoxLayer

}  //namespace vision

#endif