/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file dlcv_proc_opt.h
 * @brief This header file defines dlcv_proc_opt's interface
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-12-20
 */

#ifndef __DLCV_PROC_OPT_H__
#define __DLCV_PROC_OPT_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void  dlcv_image_togray_uc(const unsigned char*   src,
                           unsigned char*         dst,
                           int                    src_width,
                           int                    src_height,
                           int                    src_line_size,
                           int                    dst_line_size,
                           int                    src_channel,   //just 3 or 4
                           int                    bgr_or_rgb     //0 is bgr, 1 is rgb
                           );

void  dlcv_image_togray_f(const float*           src,
                          float*                 dst,
                          int                    src_width,
                          int                    src_height,
                          int                    src_line_size,
                          int                    dst_line_size,
                          int                    src_channel,   //just 3 or 4
                          int                    bgr_or_rgb     //0 is bgr, 1 is rgb
                          );

void  dlcv_resize_image_uc(const unsigned char*  src,
                           unsigned char*        dst,
                           int                   src_width,
                           int                   src_height,
                           int                   src_line_size,
                           int                   dst_width,
                           int                   dst_height,
                           int                   dst_line_size,
                           int                   channel_cnt,
                           unsigned int          alpha_flag);

void  dlcv_resize_image_float(const float*       src,
                              float*             dst,
                              int                src_width,
                              int                src_height,
                              int                src_line_size,
                              int                dst_width,
                              int                dst_height,
                              int                dst_line_size,
                              int                channel_cnt,
                              unsigned int       alpha_flag);

void  dlcv_normalize_image_uc1f1(const unsigned char*  src,
                                 float*                dst,
                                 int                   width,
                                 int                   height,
                                 int                   src_line_size,
                                 int                   dst_line_size,
                                 const float*          mean,
                                 const float*          std);

void  dlcv_normalize_image_f1f1(const float*   src,
                                float*         dst,
                                int            width,
                                int            height,
                                int            src_line_size,
                                int            dst_line_size,
                                const float*   mean,
                                const float*   std);

void  dlcv_normalize_image_uc3f3(const unsigned char*  src, 
                                 float*                dst, 
                                 int                   width, 
                                 int                   height, 
                                 int                   src_line_size, 
                                 int                   dst_line_size, 
                                 const float*          mean, 
                                 const float*          std);

void  dlcv_normalize_image_f3f3(const float*  src, 
                                float*        dst, 
                                int           width, 
                                int           height, 
                                int           src_line_size, 
                                int           dst_line_size, 
                                const float*  mean, 
                                const float*  std);

void  dlcv_normalize_image_uc4f4(const unsigned char*  src, 
                                 float*                dst, 
                                 int                   width, 
                                 int                   height, 
                                 int                   src_line_size, 
                                 int                   dst_line_size, 
                                 const float*          mean, 
                                 const float*          std);

void  dlcv_normalize_image_f4f4(const float*  src, 
                                float*        dst, 
                                int           width, 
                                int           height, 
                                int           src_line_size, 
                                int           dst_line_size, 
                                const float*  mean, 
                                const float*  std);

void  dlcv_image_to_int8_f1uc1(const float*   src, 
                               unsigned char* dst, 
                               int            width, 
                               int            height, 
                               int            src_line_size, 
                               int            dst_line_size, 
                               float          step, 
                               float          zero);

void  dlcv_image_to_int8_f3uc3(const float*   src, 
                               unsigned char* dst, 
                               int            width, 
                               int            height, 
                               int            src_line_size, 
                               int            dst_line_size, 
                               float          step, 
                               float          zero);

typedef struct tag_dlcv_sp_key_point{
    int            x ;
    int            y ;
    float          score ;
    unsigned int   reserved ;
}DLCV_SP_KEY_POINT, *PDLCV__SP_KEY_POINT;

/*
* @brief extract key points from superpoint's prob data.
*
* @param prob superpoint prob's buffer, it descript every point's score, its format is hwc
* 
* @param width prob's with.
* 
* @param height prob's height.
* 
* @param channel prob's channel
* 
* @param upsample_scale superpoint's scale, such as 8
*
* @param threshold, the threshold score
* 
* @param bkg_at_last, flag for the last channel is the back_ground
*
* @param assist_buf, the operation of nms need extra buffer, 
*                    its length should be:
*                    >= 3 * ((width * upsample_scale + 2 * upsample_scale) * (height * upsample_scale + 2 * upsample_scale)) * sizeof(float)
*
* @param output_kpts, the result will be passed into this buffer
* 
* @param output_kpts_count, the element count of output_kpts, it is a input
* 
* @return the real number of points which match threshold
*
*/
int   dlcv_super_point_extract_key_points(const float*        prob,
                                          int                 width,
                                          int                 height,
                                          int                 channel,
                                          int                 upsample_scale,
                                          float               threshold,
                                          int                 bkg_at_last,
                                          unsigned char*      assist_buf,
                                          DLCV_SP_KEY_POINT*  output_kpts,
                                          int                 output_kpts_count);

/*
* @brief normalize feature from descriptor_map.
* 
* @param feature_map superpoint's descriptor map, its format is hwc
* 
* @param descriptor_width
* 
* @param descriptor_height
*
* @param descriptor_channel
*
* @param normalize_descriptor_map normalize descriptor map, 
*                                 its element_count should be >= (descriptor_width * descriptor_height * descriptor_channel)
* 
* @return none
*
*/
void   dlcv_super_point_normalize_descriptor(const float*    descriptor_map,
                                             int             descriptor_width,
                                             int             descriptor_height,
                                             int             descriptor_channel,
                                             float*          normalize_descriptor_map);

/*
* @brief extract feature from feature_map according to key_points. 
* 
* @param key_points
*
* @param key_points_count count of key_points
*
* @param descriptor_map superpoint's descriptor map, its format is hwc
* 
* @param descriptor_width
* 
* @param descriptor_height
*
* @param descriptor_channel
*
* @upsample_scale
*
* @param descriptor_out the result will be passed into this buffer, its element_count should be >= (key_points_count * feature_channel)
* 
* @return none
*
*/
void   dlcv_super_point_extract_point_descriptor(const DLCV_SP_KEY_POINT*    key_points,
                                                 int                         key_points_count,
                                                 const float*                descriptor_map,
                                                 int                         descriptor_width,
                                                 int                         descriptor_height,
                                                 int                         descriptor_channel,
                                                 int                         upsample_scale,
                                                 float*                      descriptor_out);

#ifdef __cplusplus
}
#endif

#endif