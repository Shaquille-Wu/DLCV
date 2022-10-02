#ifndef __DLCV_PROC_WRAPPER_H__
#define __DLCV_PROC_WRAPPER_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include "dlcv_proc_opt.h"

namespace vision{

namespace proc_opt{

void dlcv_resize_image(cv::Mat const&             src, 
                       int                        dst_width,
                       int                        dst_height,
                       cv::Mat&                   dst)
{
    int     src_depth         = src.type() & CV_MAT_DEPTH_MASK;
    int     src_width         = src.cols;
    int     src_height        = src.rows;
    int     src_channel       = src.channels();
    int     cur_out_w         = dst.cols;
    int     cur_out_h         = dst.rows;
    int     cur_out_channel   = dst.channels();
    int     cur_out_depth     = dst.type() & CV_MAT_DEPTH_MASK;
    int     src_line_size     = 0;
    int     dst_line_size     = 0;
    assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);
    assert(CV_8U == src_depth || CV_32F == src_depth);

    if((true == dst.empty()) ||
       (dst_width != cur_out_w) ||
       (dst_height != cur_out_h) ||
       (src_channel != cur_out_channel) ||
       (src_depth != cur_out_depth))
    {
        if(false == dst.empty())
            dst.release();
        if(CV_32F == src_depth)
        {
            if(1 == src_channel)
                dst.create(dst_height, dst_width, CV_32FC1);
            else if(3 == src_channel)
                dst.create(dst_height, dst_width, CV_32FC3);
            else
                dst.create(dst_height, dst_width, CV_32FC4);
        }
        else
        {
            if(1 == src_channel)
                dst.create(dst_height, dst_width, CV_8UC1);
            else if(3 == src_channel)
                dst.create(dst_height, dst_width, CV_8UC3);
            else
                dst.create(dst_height, dst_width, CV_8UC4);
        }
    }

    if(CV_32F == src_depth)
    {
        cv::resize(src, dst, cv::Size(dst_width, dst_height), 0., 0, cv::INTER_LINEAR);
    }
    else
    {
        src_line_size = src.step[0];
        dst_line_size = dst.step[0];
        dlcv_resize_image_uc((const unsigned char*)(src.data),
                             (unsigned char*)(dst.data),
                             src_width,
                             src_height,
                             src_line_size,
                             dst_width,
                             dst_height,
                             dst_line_size,
                             src_channel,
                             256);
    }
}

void dlcv_normalize_image(cv::Mat const&             src, 
                          cv::Mat&                   dst, 
                          std::vector<float> const&  mean, 
                          std::vector<float> const&  std)
{
    int     i                 = 0;
    int     src_depth         = src.type() & CV_MAT_DEPTH_MASK;
    int     expect_w          = src.cols;
    int     expect_h          = src.rows;
    int     expect_channel    = src.channels();
    int     expect_depth      = CV_32F;
    int     cur_out_w         = dst.cols;
    int     cur_out_h         = dst.rows;
    int     cur_out_channel   = dst.channels();
    int     cur_out_depth     = dst.type() & CV_MAT_DEPTH_MASK;
    float   mean_val[4]       = { 0.0f, 0.0f, 0.0f, 0.0f };
    float   std_val[4]        = { 0.0f, 0.0f, 0.0f, 0.0f };

    assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);
    assert(mean.size() == 1 || mean.size() == 3 || mean.size() == 4);
    assert(std.size() == 1 || std.size() == 3 || std.size() == 4);

    if((true == dst.empty()) ||
       (expect_w != cur_out_w) ||
       (expect_h != cur_out_h) ||
       (expect_channel != cur_out_channel) ||
       (expect_depth != cur_out_depth))
    {
        if(false == dst.empty())
            dst.release();
        if(1 == expect_channel)
            dst.create(expect_h, expect_w, CV_32FC1);
        else if(3 == expect_channel)
            dst.create(expect_h, expect_w, CV_32FC3);
        else
            dst.create(expect_h, expect_w, CV_32FC4);
    }

    if(1 == mean.size())
    {
        mean_val[0] = mean[0];
        mean_val[1] = mean[0];
        mean_val[2] = mean[0];
        mean_val[3] = mean[0];
        std_val[0]  = std[0];
        std_val[1]  = std[0];
        std_val[2]  = std[0];
        std_val[3]  = std[0];
    }
    else
    {
        assert(src.channels() == 3 || src.channels() == 4);
        assert(src.channels() == mean.size());
        for(i = 0 ; i < src.channels() ; i ++)
        {
            mean_val[i] = mean[i];
            std_val[i]  = std[i];
        }
    }

    int  src_line_size = 0;
    int  dst_line_size = 0;
    if(1 == expect_channel)
    {
        if(CV_32F == src_depth)
        {
            src_line_size = src.step[0] >> 2;
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_f1f1((const float*)(src.data), 
                                      (float*)(dst.data), 
                                      src.cols, 
                                      src.rows, 
                                      src_line_size, 
                                      dst_line_size, 
                                      mean_val,
                                      std_val);
        }
        else
        {
            src_line_size = src.step[0];
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_uc1f1((const unsigned char*)(src.data), 
                                       (float*)(dst.data), 
                                       src.cols, 
                                       src.rows, 
                                       src_line_size, 
                                       dst_line_size, 
                                       mean_val,
                                       std_val);
        }
    }
    else if(3 == expect_channel)
    {
        if(CV_32F == src_depth)
        {
            src_line_size = src.step[0] >> 2;
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_f3f3((const float*)(src.data), 
                                      (float*)(dst.data), 
                                      src.cols, 
                                      src.rows, 
                                      src_line_size, 
                                      dst_line_size, 
                                      mean_val,
                                      std_val);
        }
        else
        {
            src_line_size = src.step[0];
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_uc3f3((const unsigned char*)(src.data), 
                                       (float*)(dst.data), 
                                       src.cols, 
                                       src.rows, 
                                       src_line_size, 
                                       dst_line_size, 
                                       mean_val,
                                       std_val);
        }
    }
    else
    {
        if(CV_32F == src_depth)
        {
            src_line_size = src.step[0] >> 2;
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_f4f4((const float*)(src.data), 
                                      (float*)(dst.data), 
                                      src.cols, 
                                      src.rows, 
                                      src_line_size, 
                                      dst_line_size, 
                                      mean_val,
                                      std_val);
        }
        else
        {
            src_line_size = src.step[0];
            dst_line_size = dst.step[0] >> 2;
            dlcv_normalize_image_uc4f4((const unsigned char*)(src.data), 
                                       (float*)(dst.data), 
                                       src.cols, 
                                       src.rows, 
                                       src_line_size, 
                                       dst_line_size, 
                                       mean_val,
                                       std_val);
        }
    }
}

void dlcv_image_togray(cv::Mat const& src, cv::Mat& dst, int bgr_or_rgb)
{
    int     i                 = 0;
    int     src_depth         = src.type() & CV_MAT_DEPTH_MASK;
    int     src_channel       = src.channels();
    int     expect_w          = src.cols;
    int     expect_h          = src.rows;
    int     expect_channel    = 1;
    int     cur_out_w         = dst.cols;
    int     cur_out_h         = dst.rows;
    int     cur_out_channel   = dst.channels();
    int     cur_out_depth     = dst.type() & CV_MAT_DEPTH_MASK;

    assert(src_channel == 3 || src_channel == 4);
    assert(src_depth == CV_32F || src_depth == CV_8U);

    if(true == dst.empty() || 
       src_depth != cur_out_depth ||
       expect_w != cur_out_w ||
       expect_h != cur_out_h ||
       cur_out_channel != 1)
    {
        if(false == dst.empty())
            dst.release();
        if(src_depth == CV_32F)
            dst.create(expect_h, expect_w, CV_32FC1);
        else
            dst.create(expect_h, expect_w, CV_8UC1);
    }   

    int  src_line_size = 0;
    int  dst_line_size = 0;
    if(CV_8U == src_depth)
    {
        src_line_size = src.step[0];
        dst_line_size = dst.step[0];
        dlcv_image_togray_uc((unsigned char*)(src.data), 
                             (unsigned char*)(dst.data), 
                             expect_w, 
                             expect_h,
                             src_line_size,
                             dst_line_size,
                             src_channel,
                             bgr_or_rgb);
    }
    else
    {
        src_line_size = src.step[0] >> 2;
        dst_line_size = dst.step[0] >> 2;
        dlcv_image_togray_f((float*)(src.data), 
                            (float*)(dst.data), 
                             expect_w, 
                             expect_h,
                             src_line_size,
                             dst_line_size,
                             src_channel,
                             bgr_or_rgb);
    }
}

void dlcv_image_to_int8(cv::Mat const& src, cv::Mat& dst, float step, float zero)
{
    int     src_depth         = src.type() & CV_MAT_DEPTH_MASK;
    int     src_w             = src.cols;
    int     src_h             = src.rows;
    int     src_channel       = src.channels();
    int     cur_out_w         = dst.cols;
    int     cur_out_h         = dst.rows;
    int     cur_out_channel   = dst.channels();
    int     cur_out_depth     = dst.type() & CV_MAT_DEPTH_MASK;

    assert(src_channel == 1 || src_channel == 3);
    assert(src_depth == CV_32F);

    if(true == dst.empty() || 
       src_w != cur_out_w ||
       src_h != cur_out_h ||
       cur_out_depth != CV_8U ||
       cur_out_channel != src_channel)
    {
        if(false == dst.empty())
            dst.release();
        if(1 == src_channel)
            dst.create(src_h, src_w, CV_8UC1);
        else
            dst.create(src_h, src_w, CV_8UC3);
    }  

    int  src_line_size = src.step[0] / 4;
    int  dst_line_size = dst.step[0];

    if(1 == src_channel)
        dlcv_image_to_int8_f1uc1((float*)(src.data), 
                                 (unsigned char*)(dst.data), 
                                 src_w, 
                                 src_h,
                                 src_line_size,
                                 dst_line_size,
                                 step,
                                 zero);
    else
    {
        dlcv_image_to_int8_f3uc3((float*)(src.data), 
                                 (unsigned char*)(dst.data), 
                                 src_w, 
                                 src_h,
                                 src_line_size,
                                 dst_line_size,
                                 step,
                                 zero);
    }

}

}//namespace proc_opt

}//namespace vision

#endif