#ifndef  __DETECTION_OUT_LAYER_H__
#define  __DETECTION_OUT_LAYER_H__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detection_out_layer.h
 * @brief This header file defines compact detection out layer
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-04-01
 */

#include "layer_common_def.h"

namespace vision{

/**
 * @brief DetectionOutLayer 
 * 
 */
class DetectionOutLayer
{
public:

typedef struct tag_detection_out_box{
    int        class_idx;
    float      score;    
    float      min_x;
    float      min_y;
    float      max_x;
    float      max_y;
    float      box_size;
}DetectionOutBox, *PDetectionOutBox;

    static const int     kResultItemElementCount = 7;

public:
    /**
     * @brief default constructor for DetectionOutLayer
     * 
     * @exceptsafe No throw.
     */
    DetectionOutLayer() noexcept;

    /**
     * @brief destructor
     * 
     * @exceptsafe No throw.
     */ 
    virtual ~DetectionOutLayer() noexcept
    {
        release();
    };

    /**
     * @brief Copy constructor
     *
     * @param other The other DetectionOutLayer object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutLayer(const DetectionOutLayer& other) noexcept;

    /**
     * @brief assignment constructor
     * 
     * @param other The other DetectionOutLayer object.
     * 
     * @exceptsafe No throw.
     */
    DetectionOutLayer&           operator=(const DetectionOutLayer&  other) noexcept;

    /**
     * @brief Get itself DetectionOutParam
     * 
     * @return the reference of detection_out_param_
     * 
     * @exceptsafe No throw.
     */
    const DetectionOutParam&     GetDetectionOutParam() const noexcept    { return detection_out_param_; } ;

    /**
     * @brief setup detection_out according to these parameters.
     *
     * @param prior_box_cnt prior box param.
     * 
     * @param detection_out_param parameter for detection out
     * 
     * @return setup's status, 0 is ok, other is failed
     *
     * @exceptsafe No throw.
     */
    int                          Setup(int prior_box_cnt, const DetectionOutParam& detection_out_param) noexcept;

    /**
     * @brief release all of resource in detection_out_layer
     * 
     * @return none
     *
     * @exceptsafe No throw.
     */    
    void                         Free() noexcept;

    /**
     * @brief solve result.
     *
     * @param prior_box_data prior box, the size of its buffer should be: prior_box_count * kPriorBoxElementCount * sizeof(float).
     * 
     * @param prior_box_count count of prior box
     * 
     * @param variance_data variance's data, the size of its buffer should be: prior_box_cnt * variance_size * sizeof(float).
     * 
     * @param variance_size variance's size, not the count of variance_data
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
    int                          Solve(const float*   prior_box_data,
                                       int            prior_box_count,
                                       const float*   variance_data,
                                       int            variance_step,
                                       const float*   mbox_conf_data,
                                       int            mbox_conf_data_count,
                                       const float*   mbox_loc_data,
                                       int            mbox_loc_data_count) noexcept;

    /**
     * @brief get layer's top data, including priorbox and its variance
     *
     * @return layer's top data
     *
     * @exceptsafe No throw.
     */
    const float*                 GetTopData() const noexcept                       { return top_data_; };

    /**
     * @brief get count of top data, 
     *
     * @return count of data buf
     *
     * @exceptsafe No throw.
     */   
    int                          GetTopDataCount() const noexcept                  { return top_data_count_; } ;

    /**
     * @brief get count of result data
     *
     * @return count of result data
     *
     * @exceptsafe No throw.
     */  
    int                          GetResultItemCount() const noexcept               { return result_item_count_; } ;

private:
    void                         release() noexcept;

    static float                 box_size(float x_min, float y_min, float x_max, float y_max, bool is_normalized) noexcept
    {
        if(x_max <= x_min || y_max <= y_min)
            return 0.0f;
        else
        {
            float  w = x_max - x_min;
            float  h = y_max - y_min;
            if(true == is_normalized)
                return w * h;
            else
                return (w + 1.0f) * (h + 1.0f);
        }

        return 0.0f;    
    }

    static float                 box_size(const DetectionOutBox* bbox, bool is_normalized) noexcept
    {
        return box_size(bbox->min_x, bbox->min_y, bbox->max_x, bbox->max_y, is_normalized);
    }

    static float                 compute_iou(const DetectionOutBox* bboxA, const DetectionOutBox* bboxB, bool is_normalized) noexcept;

    static int                   decode_box(const int*                target_idx,
                                            int                       target_idx_cnt,
                                            char*                     prior_box_decoded_flag,
                                            const float*              prior_box_data, 
                                            const float*              variance_data, 
                                            int                       variance_size,
                                            const float*              mbox_conf_data,
                                            int                       class_count,
                                            int                       class_idx,                                   
                                            const float*              mbox_loc_data, 
                                            DetectionOutBox*          loc_decode, 
                                            int                       code_type, 
                                            bool                      variance_in_target) noexcept;

    static int                   nms(const DetectionOutBox*  boxes_sorted, 
                                     int                     boxes_cnt, 
                                     DetectionOutBox*        boxes_nms, 
                                     float                   nms_threshold, 
                                     float                   eta) noexcept;

    static int                   cmp_score(const void* left, const void* right) noexcept
    {
        DetectionOutBox*     detection_left  = (DetectionOutBox*)left;
        DetectionOutBox*     detection_right = (DetectionOutBox*)right;
        return detection_right->score > detection_left->score ? 1 : -1;
    }

private:
    DetectionOutParam            detection_out_param_;
    float*                       top_data_;
    int                          top_data_count_;    
    int                          result_item_count_;

    char*                        prior_box_decoded_flag_;
    int*                         detection_out_box_filter_idx_;
    DetectionOutBox*             detection_out_box_classes_;
    DetectionOutBox*             detection_out_box_result_;
    int                          detection_out_box_count_;
}; // DetectionOutLayer

}  // namespace vision

#endif