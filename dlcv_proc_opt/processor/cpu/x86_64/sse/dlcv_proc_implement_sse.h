#ifndef __DLCV_PROC_IMPLEMENT_SSE_H__
#define __DLCV_PROC_IMPLEMENT_SSE_H__

#include "../../../../common/dlcv_proc_opt_com_def.h"

void image_togray_uc3_implement_sse(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef);

void image_togray_uc4_implement_sse(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef);

void image_to_int8_f1uc1_aligned16_implement_sse(float const*      src,
                                                 unsigned char*    dst,
                                                 int               src_width,
                                                 int               src_height,
                                                 int               src_line_size,
                                                 int               dst_line_size,
                                                 float             scale,
                                                 float             bias);

void image_to_int8_f1uc1_aligned32_implement_sse(float const*      src,
                                                 unsigned char*    dst,
                                                 int               src_width,
                                                 int               src_height,
                                                 int               src_line_size,
                                                 int               dst_line_size,
                                                 float             scale,
                                                 float             bias);

void data_to_int8_aligned32_implement_sse(float const*      src,
                                          unsigned char*    dst,
                                          int               data_count,
                                          float             scale,
                                          float             bias);

void image_togray_f3_implement_sse(float const*   src,
                                   float*         dst,
                                   int            src_width,
                                   int            src_height,
                                   int            src_line_size,
                                   int            dst_line_size,
                                   float const*   cvt_coef);

void image_togray_f4_implement_sse(float const*   src,
                                   float*         dst,
                                   int            src_width,
                                   int            src_height,
                                   int            src_line_size,
                                   int            dst_line_size,
                                   float const*   cvt_coef);

void resize_image_uc1_row_proc_implement_sse(unsigned char const*         src_row, 
                                             unsigned char*               dst_row, 
                                             int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                             unsigned short int const*    pos_u, 
                                             unsigned long long           pos_v1,
                                             unsigned long long           pos_v0);
#else
                                             float const*                 pos_u, 
                                             float                        pos_v);
#endif

void resize_image_uc3_row_proc_implement_sse(unsigned char const*         src_row, 
                                             unsigned char*               dst_row, 
                                             int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                             unsigned short int const*    pos_u, 
                                             unsigned long long           pos_v1,
                                             unsigned long long           pos_v0);
#else
                                             float const*                 pos_u, 
                                             float                        pos_v);
#endif

void resize_image_uc4_row_proc_alpha_fixed_implement_sse(unsigned char const*         src_row, 
                                                         unsigned char*               dst_row, 
                                                         int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                                         unsigned short int const*    pos_u, 
                                                         unsigned long long           pos_v1,
                                                         unsigned long long           pos_v0,
#else
                                                         float const*                 pos_u, 
                                                         float                        pos_v
#endif
                                                         unsigned char                alpha_value);

void resize_image_uc4_row_proc_alpha_var_implement_sse(unsigned char const*         src_row, 
                                                       unsigned char*               dst_row, 
                                                       int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                                       unsigned short int const*    pos_u, 
                                                       unsigned long long           pos_v1,
                                                       unsigned long long           pos_v0);
#else
                                                       float const*                 pos_u, 
                                                       float                        pos_v);
#endif

void resize_image_f1_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v);

void resize_image_f3_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v);

void resize_image_f4_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v);

void normalize_image_uc1f1_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         float                 mean, 
                                         float                 inv_std);

void normalize_image_uc3f3_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         const float*          mean, 
                                         const float*          inv_std);

void normalize_image_uc4f4_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         const float*          mean, 
                                         const float*          inv_std);

void normalize_image_f1f1_implement_sse(const float*    src, 
                                        float*          dst, 
                                        int             width_aligned, 
                                        int             height, 
                                        int             src_line_element, 
                                        int             dst_line_element, 
                                        float           mean, 
                                        float           inv_std);

void normalize_image_f3f3_implement_sse(const float*    src, 
                                        float*          dst, 
                                        int             width_aligned, 
                                        int             height, 
                                        int             src_line_element, 
                                        int             dst_line_element, 
                                        const float*    mean, 
                                        const float*    inv_std);

void normalize_image_f4f4_implement_sse(const float*    src, 
                                        float*          dst, 
                                        int             width_aligned, 
                                        int             height, 
                                        int             src_line_element, 
                                        int             dst_line_element, 
                                        const float*    mean, 
                                        const float*    inv_std);

int super_point_extract_flag_implement_sse(float const*     prob, 
                                           int              channel, 
                                           int*             pt_flag,
                                           float            threshold);

void normalize_super_point_feature_implement_sse(float const*    feature_map, 
                                                 int             feature_width, 
                                                 int             feature_height, 
                                                 int             feature_channel, 
                                                 float*          normalize_feature_map);

float extract_super_point_feature_implement_sse(float const*      feature_00, 
                                                float const*      feature_01, 
                                                float const*      feature_10, 
                                                float const*      feature_11, 
                                                float             u,
                                                float             v,
                                                int               feature_channel,
                                                float*            result);

void normalize_super_point_feature_in_place_implement_sse(float* result, int feature_channel, float sum_rsqrt);
#endif