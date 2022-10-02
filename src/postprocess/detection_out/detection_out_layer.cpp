/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detection_out_layer.cpp
 * @brief the implementation for detection out layer
 * @author wuxiao@ainirobot.com
 * @date 2020-04-01
 */

#include "detection_out_layer.h"
#include <string.h>
#include <math.h>
#include <algorithm>

namespace vision{

DetectionOutLayer::DetectionOutLayer() noexcept
{
    result_item_count_             = 0;
    top_data_                      = nullptr;
    top_data_count_                = 0;
    prior_box_decoded_flag_        = nullptr;
    detection_out_box_filter_idx_  = nullptr;
    detection_out_box_classes_     = nullptr;
    detection_out_box_result_      = nullptr;
    detection_out_box_count_       = 0;
}

DetectionOutLayer::DetectionOutLayer(const DetectionOutLayer& other) noexcept
{
    detection_out_param_ = other.detection_out_param_;
    result_item_count_   = other.result_item_count_;

    if(nullptr == top_data_)
        delete top_data_;
    top_data_        = nullptr;
    top_data_count_  = 0;
    if(nullptr != other.top_data_)
    {
        top_data_         = new float[other.top_data_count_];
        top_data_count_   = other.top_data_count_;
        memcpy(top_data_, other.top_data_, top_data_count_ * sizeof(float));
    }

    if(nullptr != prior_box_decoded_flag_)
        delete prior_box_decoded_flag_;
    prior_box_decoded_flag_         = nullptr;
    if(nullptr != detection_out_box_filter_idx_)
        delete detection_out_box_filter_idx_;
    detection_out_box_filter_idx_   = nullptr;
    detection_out_box_count_        = 0;
    if(nullptr != other.detection_out_box_filter_idx_)
    {
        prior_box_decoded_flag_         = new char[other.detection_out_box_count_];
        detection_out_box_filter_idx_   = new int[other.detection_out_box_count_];
        detection_out_box_count_        = other.detection_out_box_count_;
        memcpy(prior_box_decoded_flag_, other.prior_box_decoded_flag_, other.detection_out_box_count_ * sizeof(char));
        memcpy(detection_out_box_filter_idx_, other.detection_out_box_filter_idx_, other.detection_out_box_count_ * sizeof(int));
    }

    if(nullptr == detection_out_box_classes_)
        delete detection_out_box_classes_;
    detection_out_box_classes_  = nullptr;
    if(nullptr != other.detection_out_box_classes_)
    {
        detection_out_box_classes_ = new DetectionOutBox[3 * other.detection_out_box_count_];
        memcpy(detection_out_box_classes_, other.detection_out_box_classes_, 3 * other.detection_out_box_count_ * sizeof(DetectionOutBox));
    }

    if(nullptr == detection_out_box_result_)
        delete detection_out_box_result_;
    detection_out_box_result_  = nullptr;
    if(nullptr != other.detection_out_box_result_)
    {
        detection_out_box_result_ = new DetectionOutBox[3 * other.detection_out_box_count_];
        memcpy(detection_out_box_result_, other.detection_out_box_result_, 3 * other.detection_out_box_count_ * sizeof(DetectionOutBox));
    }
}

DetectionOutLayer& DetectionOutLayer::operator=(const DetectionOutLayer&  other) noexcept
{
    if(&other == this)
        return *this;
    
    detection_out_param_ = other.detection_out_param_;
    result_item_count_   = other.result_item_count_;

    if(nullptr == top_data_)
        delete top_data_;
    top_data_        = nullptr;
    top_data_count_  = 0;
    if(nullptr != other.top_data_)
    {
        top_data_         = new float[other.top_data_count_];
        top_data_count_   = other.top_data_count_;
        memcpy(top_data_, other.top_data_, top_data_count_ * sizeof(float));
    }

    if(nullptr != prior_box_decoded_flag_)
        delete prior_box_decoded_flag_;
    prior_box_decoded_flag_         = nullptr;
    if(nullptr != detection_out_box_filter_idx_)
        delete detection_out_box_filter_idx_;
    detection_out_box_filter_idx_ = nullptr;
    detection_out_box_count_      = 0;
    if(nullptr != other.detection_out_box_filter_idx_)
    {
        prior_box_decoded_flag_        = new char[other.detection_out_box_count_];
        detection_out_box_filter_idx_  = new int[other.detection_out_box_count_];
        detection_out_box_count_       = other.detection_out_box_count_;
        memcpy(prior_box_decoded_flag_, other.prior_box_decoded_flag_, other.detection_out_box_count_ * sizeof(char));
        memcpy(detection_out_box_filter_idx_, other.detection_out_box_filter_idx_, other.detection_out_box_count_ * sizeof(int));
    }

    if(nullptr == detection_out_box_classes_)
        delete detection_out_box_classes_;
    detection_out_box_classes_  = nullptr;
    if(nullptr != other.detection_out_box_classes_)
    {
        detection_out_box_classes_ = new DetectionOutBox[3 * other.detection_out_box_count_];
        memcpy(detection_out_box_classes_, other.detection_out_box_classes_, 3 * other.detection_out_box_count_ * sizeof(DetectionOutBox));
    }

    if(nullptr == detection_out_box_result_)
        delete detection_out_box_result_;
    detection_out_box_result_  = nullptr;
    if(nullptr != other.detection_out_box_result_)
    {
        detection_out_box_result_ = new DetectionOutBox[3 * other.detection_out_box_count_];
        memcpy(detection_out_box_result_, other.detection_out_box_result_, 3 * other.detection_out_box_count_ * sizeof(DetectionOutBox));
    }

    return *this;
};

int DetectionOutLayer::Setup(int prior_box_cnt, const DetectionOutParam& detection_out_param) noexcept
{

    if(detection_out_param.top_k_ <= 0)
        top_data_count_ = detection_out_param.class_count_ * prior_box_cnt * kResultItemElementCount;
    else
    {
        top_data_count_ = detection_out_param.top_k_ < prior_box_cnt ? detection_out_param.top_k_ : prior_box_cnt;
        top_data_count_ = top_data_count_ * kResultItemElementCount;
    }
    top_data_                 = new float[top_data_count_];
    memset(top_data_, 0, top_data_count_ * sizeof(float));

    prior_box_decoded_flag_           = new char[prior_box_cnt];
    detection_out_box_filter_idx_     = new int[prior_box_cnt];
    detection_out_box_classes_        = new DetectionOutBox[3 * prior_box_cnt];
    detection_out_box_result_         = new DetectionOutBox[3 * prior_box_cnt];
    detection_out_box_count_          = prior_box_cnt;
    memset(prior_box_decoded_flag_,       0, prior_box_cnt * sizeof(char));
    memset(detection_out_box_filter_idx_, 0, prior_box_cnt * sizeof(int));
    memset(detection_out_box_classes_,    0, 3 * prior_box_cnt * sizeof(DetectionOutBox));
    memset(detection_out_box_result_,     0, 3 * prior_box_cnt * sizeof(DetectionOutBox));

    detection_out_param_ = detection_out_param;

    return 0;
}

void DetectionOutLayer::Free() noexcept
{
    release();
}

int DetectionOutLayer::Solve(const float*   prior_box_data,
                             int            prior_box_count,
                             const float*   variance_data,
                             int            variance_step,
                             const float*   mbox_conf_data,
                             int            mbox_conf_data_count,
                             const float*   mbox_loc_data,
                             int            mbox_loc_data_count) noexcept
{
    int      i                     = 0;
    int      j                     = 0;
    int      result_count          = 0;
    int      all_result_count      = 0;
    int      valid_class_count     = 0;
    int      class_count           = detection_out_param_.class_count_;
    if(class_count < 1)
        class_count = 1;
    if((prior_box_count * PriorBoxParam::kPriorBoxElementCount != mbox_loc_data_count) ||
       ((prior_box_count * class_count) != mbox_conf_data_count))
       return -1;

    DetectionOutBox*  box_decode = detection_out_box_classes_;
    DetectionOutBox*  box_sort   = box_decode + prior_box_count;
    DetectionOutBox*  box_nms    = box_sort   + prior_box_count;

    memset(prior_box_decoded_flag_, 0, prior_box_count);
    for(i = 0 ; i < detection_out_param_.class_count_ ; i ++)
    {
        if(i == detection_out_param_.background_label_id_)
            continue;
        result_count = 0;
        for(j = 0 ; j < prior_box_count ; j ++)
        {
            if(mbox_conf_data[class_count * j + i] >= detection_out_param_.conf_threshold_)
            {
                detection_out_box_filter_idx_[result_count] = j;
                result_count ++;
            }
        }
        if(result_count > 0)
        {
            decode_box(detection_out_box_filter_idx_,
                       result_count,
                       prior_box_decoded_flag_,
                       prior_box_data, 
                       variance_data, 
                       variance_step, 
                       mbox_conf_data,
                       class_count,
                       i,
                       mbox_loc_data,
                       box_decode,
                       detection_out_param_.bbox_code_type_,
                       false);
            for(j = 0 ; j < result_count ; j ++)
            {
                box_sort[j]           = box_decode[detection_out_box_filter_idx_[j]];
                box_sort[j].class_idx = i;
            }
            
            qsort(box_sort, result_count, sizeof(DetectionOutBox), cmp_score);
            result_count = nms(box_sort,
                               result_count, 
                               box_nms,
                               detection_out_param_.nms_threshold_, 
                               1.0f);
            memcpy(detection_out_box_result_ + all_result_count, box_nms, result_count * sizeof(DetectionOutBox));
            all_result_count += result_count;
            valid_class_count ++;
        }
    }

    if(valid_class_count > 1)
        qsort(detection_out_box_result_, all_result_count, sizeof(DetectionOutBox), cmp_score);
    
    if(detection_out_param_.top_k_ > 0)
    {
        all_result_count = all_result_count > detection_out_param_.top_k_ ? detection_out_param_.top_k_ : all_result_count;
    }

    for(i = 0 ; i < all_result_count ; i ++)
    {
        top_data_[i * kResultItemElementCount]     = 0.0f ;
        top_data_[i * kResultItemElementCount + 1] = (float)(detection_out_box_result_[i].class_idx) ;
        top_data_[i * kResultItemElementCount + 2] = detection_out_box_result_[i].score ;
        top_data_[i * kResultItemElementCount + 3] = detection_out_box_result_[i].min_x ;
        top_data_[i * kResultItemElementCount + 4] = detection_out_box_result_[i].min_y ;
        top_data_[i * kResultItemElementCount + 5] = detection_out_box_result_[i].max_x ;
        top_data_[i * kResultItemElementCount + 6] = detection_out_box_result_[i].max_y ;
    }

    result_item_count_ = all_result_count;

    return 0;
}

void DetectionOutLayer::release() noexcept
{
    detection_out_param_.class_count_         = 1;
    detection_out_param_.background_label_id_ = 0;
    detection_out_param_.conf_threshold_      = 0.0f;
    detection_out_param_.nms_threshold_       = 0.0f;
    detection_out_param_.top_k_               = 0;
    detection_out_param_.bbox_code_type_      = CODE_TYPE_CENTER_SIZE;

    result_item_count_ = 0;
    if(nullptr != top_data_)
    {
        delete top_data_;
    }
    top_data_          = nullptr;
    top_data_count_    = 0;

    if(nullptr != prior_box_decoded_flag_)
        delete prior_box_decoded_flag_;
    prior_box_decoded_flag_ = nullptr;    
    if(nullptr != detection_out_box_filter_idx_)
        delete detection_out_box_filter_idx_;
    detection_out_box_filter_idx_ = nullptr;

    if(nullptr != detection_out_box_classes_)
        delete detection_out_box_classes_;
    detection_out_box_classes_ = nullptr;

    if(nullptr != detection_out_box_result_)
        delete detection_out_box_result_;
    detection_out_box_result_  = nullptr;        
    detection_out_box_count_   = 0;
}

float DetectionOutLayer::compute_iou(const DetectionOutBox* bboxA, const DetectionOutBox* bboxB, bool is_normalized) noexcept
{
    DetectionOutBox   intersect_bbox   = { 0 };
    float             intersect_width  = 0.0f;
    float             intersect_height = 0.0f;
    float             intersect_size   = 0.0f;
    float             iou              = 0.0f;
    if(bboxB->min_x >= bboxA->max_x || bboxB->max_x <= bboxA->min_x ||
        bboxB->min_y >= bboxA->max_y || bboxB->max_y <= bboxA->min_y)
    {
        return 0.0f;
    }

    intersect_bbox.min_x = bboxA->min_x > bboxB->min_x ? bboxA->min_x : bboxB->min_x;
    intersect_bbox.min_y = bboxA->min_y > bboxB->min_y ? bboxA->min_y : bboxB->min_y;
    intersect_bbox.max_x = bboxA->max_x < bboxB->max_x ? bboxA->max_x : bboxB->max_x;
    intersect_bbox.max_y = bboxA->max_y < bboxB->max_y ? bboxA->max_y : bboxB->max_y;

    intersect_width  = intersect_bbox.max_x - intersect_bbox.min_x;
    intersect_height = intersect_bbox.max_y - intersect_bbox.min_y;  

    if(false == is_normalized)
    {
        intersect_width  += 1.0f;
        intersect_height += 1.0f;
    }

    intersect_size  = intersect_width * intersect_height;

    iou             = intersect_size / (bboxA->box_size + bboxB->box_size - intersect_size);

    return iou;
}

int DetectionOutLayer::decode_box(const int*            target_idx,
                                  int                   target_idx_cnt,
                                  char*                 prior_box_decoded_flag,
                                  const float*          prior_box_data, 
                                  const float*          variance_data, 
                                  int                   variance_size,                                  
                                  const float*          mbox_conf_data,
                                  int                   class_count,
                                  int                   class_idx,
                                  const float*          mbox_loc_data, 
                                  DetectionOutBox*      loc_decode, 
                                  int                   code_type, 
                                  bool                  variance_in_target) noexcept
{
    int i = 0;
    if(CODE_TYPE_CORNER == code_type)
    {
        return 0;
    }

    else if(CODE_TYPE_CENTER_SIZE == code_type)
    {
        for(i = 0 ; i < target_idx_cnt ; i ++)
        {
            int     box_idx         = target_idx[i];
            if(1 == prior_box_decoded_flag[box_idx])
            {
                loc_decode[box_idx].score = mbox_conf_data[class_count * box_idx + class_idx];
                continue;
            }

            float   prior_width     = prior_box_data[4 * box_idx + 2] - prior_box_data[4 * box_idx];
            float   prior_height    = prior_box_data[4 * box_idx + 3] - prior_box_data[4 * box_idx + 1];
            float   prior_center_x  = 0.5f * (prior_box_data[4 * box_idx]     + prior_box_data[4 * box_idx + 2]);
            float   prior_center_y  = 0.5f * (prior_box_data[4 * box_idx + 1] + prior_box_data[4 * box_idx + 3]);

            float decode_bbox_center_x = 0.0f, decode_bbox_center_y = 0.0f;
            float decode_bbox_width    = 0.0f, decode_bbox_height   = 0.0f;
            if (true == variance_in_target)
            {
                // variance is encoded in target, we simply need to retore the offset
                // predictions.
                decode_bbox_center_x = mbox_loc_data[4 * box_idx]     * prior_width  + prior_center_x;
                decode_bbox_center_y = mbox_loc_data[4 * box_idx + 1] * prior_height + prior_center_y;
                decode_bbox_width    = expf(mbox_loc_data[4 * box_idx + 2]) * prior_width;
                decode_bbox_height   = expf(mbox_loc_data[4 * box_idx + 3]) * prior_height;
            }
            else 
            {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = variance_data[4 * box_idx]          * mbox_loc_data[4 * box_idx]     * prior_width + prior_center_x;
                decode_bbox_center_y = variance_data[4 * box_idx + 1]      * mbox_loc_data[4 * box_idx + 1] * prior_height + prior_center_y;
                decode_bbox_width    = expf(variance_data[4 * box_idx + 2] * mbox_loc_data[4 * box_idx + 2]) * prior_width;
                decode_bbox_height   = expf(variance_data[4 * box_idx + 3] * mbox_loc_data[4 * box_idx + 3]) * prior_height;
            }

            loc_decode[box_idx].score       = mbox_conf_data[class_count * box_idx + class_idx];
            loc_decode[box_idx].min_x       = decode_bbox_center_x - 0.5f * decode_bbox_width;
            loc_decode[box_idx].min_y       = decode_bbox_center_y - 0.5f * decode_bbox_height;
            loc_decode[box_idx].max_x       = decode_bbox_center_x + 0.5f * decode_bbox_width;
            loc_decode[box_idx].max_y       = decode_bbox_center_y + 0.5f * decode_bbox_height;
            loc_decode[box_idx].box_size    = box_size(loc_decode[box_idx].min_x, loc_decode[box_idx].min_y, loc_decode[box_idx].max_x, loc_decode[box_idx].max_y, true);
            prior_box_decoded_flag[box_idx] = 1;
        }
    }
    else if(CODE_TYPE_CORNER_SIZE == code_type)
    {

    }
    else
        return -1;
    
    return 0;
}

int DetectionOutLayer::nms(const DetectionOutBox*  boxes_sorted, 
                           int                     boxes_cnt, 
                           DetectionOutBox*        boxes_nms, 
                           float                   nms_threshold, 
                           float                   eta) noexcept
{
    int      i                   = 0;
    int      j                   = 0;
    int      result_cnt          = 0;
    float    adaptive_threshold  = nms_threshold;
    for(i = 0 ; i < boxes_cnt ; i ++)
    {
        int  is_kept = 1;
        for(j = 0; j < result_cnt ; j ++)
        {
            if(1 == is_kept)
            {
                float   iou = compute_iou(boxes_sorted + i, boxes_nms + j, 1);
                is_kept     = (iou <= adaptive_threshold ? 1 : 0);
            }
            else
                break;
        }

        if(1 == is_kept)
        {
            memcpy(boxes_nms + result_cnt, boxes_sorted + i, sizeof(DetectionOutBox));
            result_cnt ++;
        }

        if(1 == is_kept && eta < 1.0f && adaptive_threshold > 0.5f)
        {
            adaptive_threshold *= eta;
        }
    }

    return result_cnt;
}

} //namespace vision