#include <emmintrin.h>
#include "../../../normalize_image/normalize_image.h"
#include "../../../resize_image/resize_image.h"
#include "../../../resize_image/resize_image_uc_row_proc_implement.h"
#include "../../../super_point/super_point.h"
#include "../../../to_gray/to_gray.h"
#include "./sse/dlcv_proc_implement_sse.h"
#include "./avx/dlcv_proc_implement_avx.h"

void image_togray_uc3_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef)
{
    image_togray_uc3_implement_sse(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}

void image_togray_uc4_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef)
{
    image_togray_uc4_implement_avx(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}

void image_togray_f3_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef)
{
    image_togray_f3_implement_avx(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}

void image_togray_f4_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef)
{
    image_togray_f4_implement_avx(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}

void image_to_int8_f1uc1_implement(float const*      src,
                                   unsigned char*    dst,
                                   int               src_width,
                                   int               src_height,
                                   int               src_line_size,
                                   int               dst_line_size,
                                   float             scale,
                                   float             bias)
{
    if(0 == (src_width & 0x1F))
        image_to_int8_f1uc1_aligned32_implement_avx(src, dst, src_width, src_height, src_line_size, dst_line_size, scale, bias);
    else
        image_to_int8_f1uc1_aligned16_implement_avx(src, dst, src_width, src_height, src_line_size, dst_line_size, scale, bias);
}

void data_to_int8_implement(float const*      src,
                            unsigned char*    dst,
                            int               data_count,
                            float             scale,
                            float             bias)
{
    data_to_int8_aligned32_implement_avx(src, dst, data_count, scale, bias);
}

static void CopyDataWithPosF1(float const*   src, 
                              float*         dst, 
                              int            width, 
                              int            src_y0, 
                              int            src_y1, 
                              int const*     pos_x0,
                              int            pos_x1_valid_start,
                              int            pos_x1_valid_end)
{
    int j = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        unsigned long long x0                   = *((unsigned int*)(src + src_y0));
        unsigned long long x1                   = *((unsigned int*)(src + src_y1));
        *(unsigned long long*)(dst + 4 * j)     = x0 | (x0 << 32);
        *(unsigned long long*)(dst + 4 * j + 2) = x1 | (x1 << 32);
    }
    for(; j <= pos_x1_valid_end ; j ++)
    {
        int          src_x_0                     = pos_x0[j];
        unsigned long long x0                    = *(unsigned long long*)(src + src_y0 + src_x_0);
        unsigned long long x1                    = *(unsigned long long*)(src + src_y1 + src_x_0);
        *(unsigned long long*)(dst + 4 * j)      = x0;
        *(unsigned long long*)(dst + 4 * j + 2)  = x1;
    }

    for(; j < width ; j ++)
    {
        int          src_x_0                    = pos_x0[j];
        unsigned long long x0                   = *((unsigned int*)(src + src_y0 + src_x_0));
        unsigned long long x1                   = *((unsigned int*)(src + src_y1 + src_x_0));
        *(unsigned long long*)(dst + 4 * j)     = x0 | (x0 << 32);
        *(unsigned long long*)(dst + 4 * j + 2) = x1 | (x1 << 32);
    }
}

static void CopyDataWithPosF3(float const*   src, 
                              float*         dst, 
                              int            width, 
                              int            src_y0, 
                              int            src_y1, 
                              int const*     pos_x0,
                              int            pos_x1_valid_start,
                              int            pos_x1_valid_end)
{
    int  j       = 0;
    int  cur_pos = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        unsigned long long x0                     = *((unsigned long long*)(src + src_y0));
        unsigned long long x1                     = *((unsigned int*      )(src + src_y0 + 2));
        unsigned long long x2                     = *((unsigned long long*)(src + src_y1));
        unsigned long long x3                     = *((unsigned int*      )(src + src_y1 + 2));
        *(unsigned long long*)(dst + cur_pos)      = x0;
        *(unsigned long long*)(dst + cur_pos +  2) = x1;
        *(unsigned long long*)(dst + cur_pos +  4) = x0;
        *(unsigned long long*)(dst + cur_pos +  6) = x1;
        *(unsigned long long*)(dst + cur_pos +  8) = x2;
        *(unsigned long long*)(dst + cur_pos + 10) = x3;
        *(unsigned long long*)(dst + cur_pos + 12) = x2;
        *(unsigned long long*)(dst + cur_pos + 14) = x3;
        cur_pos += 16;
    }
    for(; j <= pos_x1_valid_end - 1; j ++)
    {
        int          src_x_0                      = pos_x0[j];
        __m128       m0 = _mm_loadu_ps(src + src_y0 + 3 * src_x_0);
        __m128       m1 = _mm_loadu_ps(src + src_y0 + 3 * src_x_0 + 3);
        __m128       m2 = _mm_loadu_ps(src + src_y1 + 3 * src_x_0);
        __m128       m3 = _mm_loadu_ps(src + src_y1 + 3 * src_x_0 + 3);
        _mm_storeu_ps(dst + cur_pos,      m0);
        _mm_storeu_ps(dst + cur_pos +  4, m1);
        _mm_storeu_ps(dst + cur_pos +  8, m2);
        _mm_storeu_ps(dst + cur_pos + 12, m3);
        cur_pos += 16;
    }

    if(j <= pos_x1_valid_end)
    {
        int          src_x_0                      = pos_x0[j];
        unsigned long long x0                     = *((unsigned long long*)(src + src_y0 + 3 * src_x_0));
        unsigned long long x1                     = *((unsigned int*      )(src + src_y0 + 3 * src_x_0 + 2));
        unsigned long long x2                     = *((unsigned long long*)(src + src_y0 + 3 * src_x_0 + 3));
        unsigned long long x3                     = *((unsigned int*      )(src + src_y0 + 3 * src_x_0 + 5));
        unsigned long long x4                     = *((unsigned long long*)(src + src_y1 + 3 * src_x_0));
        unsigned long long x5                     = *((unsigned int*      )(src + src_y1 + 3 * src_x_0 + 2));
        unsigned long long x6                     = *((unsigned long long*)(src + src_y1 + 3 * src_x_0 + 3));
        unsigned long long x7                     = *((unsigned int*      )(src + src_y1 + 3 * src_x_0 + 5));
        *(unsigned long long*)(dst + cur_pos)      = x0;
        *(unsigned long long*)(dst + cur_pos +  2) = x1;
        *(unsigned long long*)(dst + cur_pos +  4) = x2;
        *(unsigned long long*)(dst + cur_pos +  6) = x3;
        *(unsigned long long*)(dst + cur_pos +  8) = x4;
        *(unsigned long long*)(dst + cur_pos + 10) = x5;
        *(unsigned long long*)(dst + cur_pos + 12) = x6;
        *(unsigned long long*)(dst + cur_pos + 14) = x7;
        cur_pos += 16;
        j++;
    }

    for(; j < width ; j ++)
    {
        int          src_x_0                      = pos_x0[j];
        unsigned long long x0                     = *((unsigned long long*)(src + src_y0 + 3 * src_x_0));
        unsigned long long x1                     = *((unsigned int*      )(src + src_y0 + 3 * src_x_0 + 2));
        unsigned long long x2                     = *((unsigned long long*)(src + src_y1 + 3 * src_x_0));
        unsigned long long x3                     = *((unsigned int*      )(src + src_y1 + 3 * src_x_0 + 2));
        *(unsigned long long*)(dst + cur_pos)      = x0;
        *(unsigned long long*)(dst + cur_pos +  2) = x1;
        *(unsigned long long*)(dst + cur_pos +  4) = x0;
        *(unsigned long long*)(dst + cur_pos +  6) = x1;
        *(unsigned long long*)(dst + cur_pos +  8) = x2;
        *(unsigned long long*)(dst + cur_pos + 10) = x3;
        *(unsigned long long*)(dst + cur_pos + 12) = x2;
        *(unsigned long long*)(dst + cur_pos + 14) = x3;
        cur_pos += 16;
    }
}

static void CopyDataWithPosF4(float const*   src, 
                              float*         dst, 
                              int            width, 
                              int            src_y0, 
                              int            src_y1, 
                              int const*     pos_x0,
                              int            pos_x1_valid_start,
                              int            pos_x1_valid_end)
{
    int  j       = 0;
    int  cur_pos = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        __m128       m0 = _mm_loadu_ps(src + src_y0);
        __m128       m1 = _mm_loadu_ps(src + src_y1);
        _mm_store_ps(dst + cur_pos,      m0);
        _mm_store_ps(dst + cur_pos +  4, m0);
        _mm_store_ps(dst + cur_pos +  8, m1);
        _mm_store_ps(dst + cur_pos + 12, m1);
        cur_pos += 16;
    }
    for(; j <= pos_x1_valid_end; j ++)
    {
        int          src_x_0                      = pos_x0[j];
        __m128       m0 = _mm_loadu_ps(src + src_y0 + 4 * src_x_0);
        __m128       m1 = _mm_loadu_ps(src + src_y0 + 4 * src_x_0 + 4);
        __m128       m2 = _mm_loadu_ps(src + src_y1 + 4 * src_x_0);
        __m128       m3 = _mm_loadu_ps(src + src_y1 + 4 * src_x_0 + 4);
        _mm_store_ps(dst + cur_pos,      m0);
        _mm_store_ps(dst + cur_pos +  4, m1);
        _mm_store_ps(dst + cur_pos +  8, m2);
        _mm_store_ps(dst + cur_pos + 12, m3);
        cur_pos += 16;
    }

    for(; j < width ; j ++)
    {
        int          src_x_0                      = pos_x0[j];
        __m128       m0 = _mm_loadu_ps(src + src_y0 + 4 * src_x_0);
        __m128       m1 = _mm_loadu_ps(src + src_y1 + 4 * src_x_0);
        _mm_store_ps(dst + cur_pos,      m0);
        _mm_store_ps(dst + cur_pos +  4, m0);
        _mm_store_ps(dst + cur_pos +  8, m1);
        _mm_store_ps(dst + cur_pos + 12, m1);
        cur_pos += 16;
    }
}

void resize_image_uc1_row_proc_implement(unsigned char const*         src_row, 
                                         unsigned char*               dst_row, 
                                         int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                         unsigned short int const*    pos_u, 
                                         unsigned long long           pos_v1,
                                         unsigned long long           pos_v0)
#else
                                         float const*                 pos_u, 
                                         float                        pos_v)
#endif
{
        resize_image_uc1_row_proc_implement_avx(src_row,
                                                dst_row,
                                                dst_width,
                                                pos_u,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                pos_v1,
                                                pos_v0);
#else
                                                pos_v);
#endif
}

void resize_image_uc3_row_proc_implement(unsigned char const*         src_row, 
                                         unsigned char*               dst_row, 
                                         int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                         unsigned short int const*    pos_u, 
                                         unsigned long long           pos_v1,
                                         unsigned long long           pos_v0)
#else
                                         float const*                 pos_u, 
                                         float                        pos_v)
#endif
{
        resize_image_uc3_row_proc_implement_avx(src_row,
                                                dst_row,
                                                dst_width,
                                                pos_u,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                pos_v1,
                                                pos_v0);
#else
                                                pos_v);
#endif
}

void resize_image_uc4_row_proc_alpha_fixed_implement(unsigned char const*         src_row, 
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
                                                     unsigned char                alpha_value)
{
    resize_image_uc4_row_proc_alpha_fixed_implement_avx(src_row,
                                                        dst_row,
                                                        dst_width,
                                                        pos_u,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                        pos_v1,
                                                        pos_v0,
#else
                                                        pos_v,
#endif
                                                        (unsigned char)alpha_value);
}

void resize_image_uc4_row_proc_alpha_var_implement(unsigned char const*         src_row, 
                                                   unsigned char*               dst_row, 
                                                   int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                                   unsigned short int const*    pos_u, 
                                                   unsigned long long           pos_v1,
                                                   unsigned long long           pos_v0)
#else
                                                   float const*                 pos_u, 
                                                   float                        pos_v)
#endif
{
    resize_image_uc4_row_proc_alpha_var_implement_avx(src_row,
                                                      dst_row,
                                                      dst_width,
                                                      pos_u,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                      pos_v1,
                                                      pos_v0);
#else
                                                      pos_v);
#endif
}

void resize_image_f1_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf)
{
    int    i                   = 0;
    for (i = 0; i < dst_height; i++)
    {
        int        src_y_start0            = pos_y_0_suppress[i] * src_line_size;
        int        src_y_start1            = pos_y_1_suppress[i] * src_line_size;
        float*     src_row_extract_buf_ptr = src_row_extract_buf;
        float*     dst_ptr                 = dst + i * dst_line_size;
        CopyDataWithPosF1(src, 
                          src_row_extract_buf_ptr, 
                          dst_width_aligned, 
                          src_y_start0, 
                          src_y_start1, 
                          pos_x_0_suppress, 
                          pos_x_1_limit[0],
                          pos_x_1_limit[1]);
        resize_image_f1_row_proc_implement_avx(src_row_extract_buf_ptr,
                                               dst_ptr,
                                               dst_width_aligned,
                                               pos_x,
                                               pos_y[i]);
    }
}

void resize_image_f3_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf)
{
    int    i                   = 0;
    for (i = 0; i < dst_height; i++)
    {
        int        src_y_start0            = pos_y_0_suppress[i] * src_line_size;
        int        src_y_start1            = pos_y_1_suppress[i] * src_line_size;
        float*     src_row_extract_buf_ptr = src_row_extract_buf;
        float*     dst_ptr                 = dst + i * dst_line_size;
        CopyDataWithPosF3(src, 
                          src_row_extract_buf_ptr, 
                          dst_width_aligned, 
                          src_y_start0, 
                          src_y_start1, 
                          pos_x_0_suppress, 
                          pos_x_1_limit[0],
                          pos_x_1_limit[1]);
        resize_image_f3_row_proc_implement_avx(src_row_extract_buf_ptr,
                                               dst_ptr,
                                               dst_width_aligned,
                                               pos_x,
                                               pos_y[i]);
    }
}

void resize_image_f4_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf)
{
    int    i                   = 0;
    for (i = 0; i < dst_height; i++)
    {
        int        src_y_start0            = pos_y_0_suppress[i] * src_line_size;
        int        src_y_start1            = pos_y_1_suppress[i] * src_line_size;
        float*     src_row_extract_buf_ptr = src_row_extract_buf;
        float*     dst_ptr                 = dst + i * dst_line_size;
        CopyDataWithPosF4(src, 
                          src_row_extract_buf_ptr, 
                          dst_width_aligned, 
                          src_y_start0, 
                          src_y_start1, 
                          pos_x_0_suppress, 
                          pos_x_1_limit[0],
                          pos_x_1_limit[1]);
        resize_image_f4_row_proc_implement_avx(src_row_extract_buf_ptr,
                                               dst_ptr,
                                               dst_width_aligned,
                                               pos_x,
                                               pos_y[i]);
    }
}

void normalize_image_uc1f1_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     float                  mean, 
                                     float                  inv_std)
{
    normalize_image_uc1f1_implement_avx(src, 
                                        dst, 
                                        width_aligned, 
                                        height, 
                                        src_line_element, 
                                        dst_line_element, 
                                        mean, 
                                        inv_std);
}

void normalize_image_uc3f3_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     const float*           mean, 
                                     const float*           inv_std)
{
    normalize_image_uc3f3_implement_avx(src, 
                                        dst, 
                                        width_aligned, 
                                        height, 
                                        src_line_element, 
                                        dst_line_element, 
                                        mean, 
                                        inv_std);
}

void normalize_image_uc4f4_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     const float*           mean, 
                                     const float*           inv_std)
{
    normalize_image_uc4f4_implement_avx(src, 
                                        dst, 
                                        width_aligned, 
                                        height, 
                                        src_line_element, 
                                        dst_line_element, 
                                        mean, 
                                        inv_std);
}

void normalize_image_f1f1_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    float          mean, 
                                    float          inv_std)
{
    normalize_image_f1f1_implement_avx(src, 
                                       dst, 
                                       width_aligned, 
                                       height, 
                                       src_line_element, 
                                       dst_line_element, 
                                       mean, 
                                       inv_std);
}

void normalize_image_f3f3_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    const float*   mean, 
                                    const float*   inv_std)
{
    normalize_image_f3f3_implement_avx(src, 
                                       dst, 
                                       width_aligned, 
                                       height, 
                                       src_line_element, 
                                       dst_line_element, 
                                       mean, 
                                       inv_std);
}

void normalize_image_f4f4_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    const float*   mean, 
                                    const float*   inv_std)
{
    normalize_image_f4f4_implement_avx(src, 
                                       dst, 
                                       width_aligned, 
                                       height, 
                                       src_line_element, 
                                       dst_line_element, 
                                       mean, 
                                       inv_std);
}

int super_point_extract_flag_implement(float const* prob, int channel, int* pt_flag, float threshold)
{
    return super_point_extract_flag_implement_avx(prob, 
                                                  channel, 
                                                  pt_flag,
                                                  threshold);
}

void normalize_super_point_feature_implement(float const*    feature_map, 
                                             int             feature_width, 
                                             int             feature_height, 
                                             int             feature_channel, 
                                             float*          normalize_feature_map)
{
    normalize_super_point_feature_implement_sse(feature_map, 
                                                feature_width, 
                                                feature_height, 
                                                feature_channel, 
                                                normalize_feature_map);
}

float extract_super_point_feature_implement(float const*      feature_00, 
                                            float const*      feature_01, 
                                            float const*      feature_10, 
                                            float const*      feature_11, 
                                            float             u,
                                            float             v,
                                            int               feature_channel,
                                            float*            result)
{
    return extract_super_point_feature_implement_avx(feature_00, 
                                                     feature_01, 
                                                     feature_10, 
                                                     feature_11, 
                                                     u,
                                                     v,
                                                     feature_channel,
                                                     result);
}

void super_point_normalize_feature_in_place_implement(float* result, int feature_channel, float sum_rsqrt)
{
    normalize_super_point_feature_in_place_implement_avx(result, feature_channel, sum_rsqrt);
}

void super_point_nms_map_8x_implement(unsigned short int const*  pt_list, 
                                      int                        pt_cnt, 
                                      unsigned char*             nms_mask_map, 
                                      int                        map_w)
{
    int  i = 0, j = 0;
    int       clean_size = 17;
    __m128i   zero       = _mm_set1_epi32(0);
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        unsigned int  xy   = *((unsigned int*)pt_list) ;
        unsigned int  x    = xy & 0x0000FFFF;
        unsigned int  y    = (xy & 0xFFFF0000) >> 16;
        unsigned int  pos  = (y + 8) * map_w + x + 8;
        unsigned char flag = nms_mask_map[pos];
        if(1 == flag)
        {
            unsigned int  clean_y = y ;
            for(j = 0 ; j < clean_size ; j ++ )
            {
                unsigned int  clean_pos = clean_y * map_w + x ;
                _mm_storeu_si128((__m128i*)(nms_mask_map + clean_pos), zero);
                nms_mask_map[clean_pos + 16] = 0;
                clean_y ++;
            }
            nms_mask_map[pos] = 255;
        }
        pt_list += 4;
    }
}