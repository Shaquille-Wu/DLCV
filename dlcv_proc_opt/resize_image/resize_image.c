#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../dlcv_proc_opt.h"
#include "./resize_image.h"
#include "../common/memory_op.h"

static const unsigned int  FIXPT_BITS    = 11;
static const unsigned int  FIXPT_ONE_VAL = 2048;//(1 << FIXPT_BITS);

static int  FloorFloat(float value)
{
    int i = (int)value;
    return i - (i > value);
}

static void caculate_pos_float(int src_len, int dst_len, float* uv, int* int_pos_0, int* int_pos_1, float ratio, int is_channel_3)
{
    int   i   = 0;
    int   j   = 0;
    int   max = src_len - 1;
    if(0 == is_channel_3)
    {
        for(i = 0 ; i < dst_len ; i ++)
        {
            uv[i]          = ((float)i) * ratio;
            int_pos_0[i]   = (int)(uv[i]);
            int_pos_1[i]   = (int)(uv[i]) + 1;
            uv[i]          = uv[i] - (float)(int_pos_0[i]);
            int_pos_0[i]   = int_pos_0[i] > max ? max : int_pos_0[i];
            int_pos_0[i]   = int_pos_0[i] <   0 ?   0 : int_pos_0[i];
            int_pos_1[i]   = int_pos_1[i] > max ? max : int_pos_1[i];
            int_pos_1[i]   = int_pos_1[i] <   0 ?   0 : int_pos_1[i];
        }
    }
    else
    {
        for(i = 0 ; i < dst_len ; i ++)
        {
            uv[i * 3]      = ((float)i) * ratio;
            int_pos_0[i]   = (int)(uv[i * 3]);
            int_pos_1[i]   = (int)(uv[i * 3]) + 1;
            uv[i * 3]      = uv[i * 3] - (float)(int_pos_0[i]);
            uv[i * 3 + 1]  = uv[i * 3];
            uv[i * 3 + 2]  = uv[i * 3];
            int_pos_0[i]   = int_pos_0[i] > max ? max : int_pos_0[i];
            int_pos_0[i]   = int_pos_0[i] <   0 ?   0 : int_pos_0[i];
            int_pos_1[i]   = int_pos_1[i] > max ? max : int_pos_1[i];
            int_pos_1[i]   = int_pos_1[i] <   0 ?   0 : int_pos_1[i];
        }
    }
}

static void caculate_pos_fixpt_opencv(int src_len, int dst_len, unsigned short* uv, int* int_pos_0, int* int_pos_1, float ratio)
{
    int                 i      = 0;
    int                 max    = src_len - 1;
    float               uv_f   = 0.0f;
    unsigned short int  uv_val = 0;
    for(i = 0 ; i < dst_len ; i ++)
    {
        uv_f           = ((float)i + 0.5f) * ratio - 0.5f;
        int_pos_0[i]   = FloorFloat(uv_f);
        int_pos_1[i]   = int_pos_0[i] + 1;
        uv_f           = (uv_f - (float)(int_pos_0[i]));
        uv_val         = (unsigned short int)(uv_f * FIXPT_ONE_VAL + 0.5f);
        uv[2 * i]      = FIXPT_ONE_VAL - uv_val;
        uv[2 * i + 1]  = uv_val;
        int_pos_0[i]   = int_pos_0[i] > max ? max : int_pos_0[i];
        int_pos_0[i]   = int_pos_0[i] <   0 ?   0 : int_pos_0[i];
        int_pos_1[i]   = int_pos_1[i] > max ? max : int_pos_1[i];
        int_pos_1[i]   = int_pos_1[i] <   0 ?   0 : int_pos_1[i];
    }
}

static void caculate_pos_float_opencv(int src_len, int dst_len, float* uv, int* int_pos_0, int* int_pos_1, float ratio)
{
    int   i   = 0;
    int   max = src_len - 1;
    //double d_ratio = (double)dst_len/(double)src_len;
    //double ratio_inv = 1.0 / d_ratio;
    for(i = 0 ; i < dst_len ; i ++)
    {
        uv[i]          = ((float)i + 0.5f) * ratio - 0.5;
        //uv[i]          = ((double)i + 0.5) * ratio_inv - 0.5;
        int_pos_0[i]   = FloorFloat(uv[i]);
        int_pos_1[i]   = int_pos_0[i] + 1;
        uv[i]          = (uv[i] - (float)(int_pos_0[i]));
        int_pos_0[i]   = int_pos_0[i] > max ? max : int_pos_0[i];
        int_pos_0[i]   = int_pos_0[i] <   0 ?   0 : int_pos_0[i];
        int_pos_1[i]   = int_pos_1[i] > max ? max : int_pos_1[i];
        int_pos_1[i]   = int_pos_1[i] <   0 ?   0 : int_pos_1[i];
    }
}

static void resize_image_uc1(const unsigned char*        src,
                             unsigned char*              dst,
                             int                         src_line_size,
                             int                         dst_width,
                             int                         dst_height,
                             int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_x,
#else
                             const float*                pos_x,
#endif
                             const int*                  pos_x_0_suppress,
                             const int*                  pos_x_1_suppress,
                             const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_y,
#else
                             const float*                pos_y,
#endif
                             const int*                  pos_y_0_suppress,
                             const int*                  pos_y_1_suppress,
                             unsigned char*              src_row_extract_buf)
{
    int width_aligned      = RESIZE_UC1_PROC_ALIGNED * (dst_width / RESIZE_UC1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_uc1_implement(src, 
                                   dst, 
                                   src_line_size, 
                                   width_aligned, 
                                   dst_height, 
                                   dst_line_size, 
                                   pos_x, 
                                   pos_x_0_suppress,
                                   pos_x_1_limit,
                                   pos_y,
                                   pos_y_0_suppress,
                                   pos_y_1_suppress,
                                   src_row_extract_buf);
    }
#ifdef RESIZE_UC_USE_FIXED_PT
    for(i = 0 ; i < dst_height ; i ++)
    {
        unsigned int   v1            = pos_y[2 * i];
        unsigned int   v0            = pos_y[2 * i + 1];
        int            pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int            pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            unsigned int   u1          = pos_x[2 * j];
            unsigned int   u0          = pos_x[2 * j + 1];
            unsigned int   src_cur_0   = src[pos_y_0_start + pos_x_0_suppress[j]];
            unsigned int   src_cur_1   = src[pos_y_0_start + pos_x_1_suppress[j]];
            unsigned int   src_cur_2   = src[pos_y_1_start + pos_x_0_suppress[j]];
            unsigned int   src_cur_3   = src[pos_y_1_start + pos_x_1_suppress[j]];
            /*
            unsigned int   res         = (((((src_cur_0 * u1 +  src_cur_1 * u0) >> 4) * v1) >> 16) +
                                          ((((src_cur_2 * u1 +  src_cur_3 * u0) >> 4) * v0) >> 16)
                                          + 2) >> 2;
            */
            unsigned int   res         = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + j] = res > 255 ? 255 : res;
        }
    }
#else
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + pos_x_1_suppress[j]];
            float        res   = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + j] = res > 255 ? 255 : ((unsigned char)res);
        }
    }
#endif
}

static void resize_image_uc3(const unsigned char*        src,
                             unsigned char*              dst,
                             int                         src_line_size,
                             int                         dst_width,
                             int                         dst_height,
                             int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_x,
#else
                             const float*                pos_x,
#endif
                             const int*                  pos_x_0_suppress,
                             const int*                  pos_x_1_suppress,
                             const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_y,
#else
                             const float*                pos_y,
#endif
                             const int*                  pos_y_0_suppress,
                             const int*                  pos_y_1_suppress,
                             unsigned char*              src_row_extract_buf)
{
    int width_aligned      = RESIZE_UC1_PROC_ALIGNED * (dst_width / RESIZE_UC1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_uc3_implement(src, 
                                   dst, 
                                   src_line_size, 
                                   width_aligned, 
                                   dst_height, 
                                   dst_line_size, 
                                   pos_x, 
                                   pos_x_0_suppress,
                                   pos_x_1_limit,
                                   pos_y,
                                   pos_y_0_suppress,
                                   pos_y_1_suppress,
                                   src_row_extract_buf);
    }

#ifdef RESIZE_UC_USE_FIXED_PT
    for(i = 0 ; i < dst_height ; i ++)
    {
        unsigned int   v1            = pos_y[2 * i];
        unsigned int   v0            = pos_y[2 * i + 1];
        int            pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int            pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            unsigned int   res0        = 0;
            unsigned int   res1        = 0;
            unsigned int   res2        = 0;
            unsigned int   u1          = pos_x[2 * j];
            unsigned int   u0          = pos_x[2 * j + 1];
            unsigned int   src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j]];
            unsigned int   src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j]];
            unsigned int   src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j]];
            unsigned int   src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j]];
            /*
            unsigned int   res         = (((((src_cur_0 * u1 +  src_cur_1 * u0) >> 4) * v1) >> 16) +
                                          ((((src_cur_2 * u1 +  src_cur_3 * u0) >> 4) * v0) >> 16)
                                          + 2) >> 2;
            */
            res0                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 3 * j] = res0 > 255 ? 255 : res0;

            src_cur_0                  = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_1                  = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 1];
            src_cur_2                  = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_3                  = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 1];
            res1                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 3 * j + 1] = res1 > 255 ? 255 : res1;
            src_cur_0                  = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_1                  = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 2];
            src_cur_2                  = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_3                  = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 2];
            res2                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 3 * j + 2] = res2 > 255 ? 255 : res2;
        }
    }
#else
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j]];
            float  res0, res1, res2;
            res0               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            src_cur_0          = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_1          = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 1];
            src_cur_2          = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_3          = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 1];
            res1               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            src_cur_0          = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_1          = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 2];
            src_cur_2          = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_3          = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 2];
            res2               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 3 * j]     = res0 > 255 ? 255 : ((unsigned char)res0);
            dst[i * dst_line_size + 3 * j + 1] = res1 > 255 ? 255 : ((unsigned char)res1);
            dst[i * dst_line_size + 3 * j + 2] = res2 > 255 ? 255 : ((unsigned char)res2);
        }
    }
#endif
}

static void resize_image_uc4(const unsigned char*        src,
                             unsigned char*              dst,
                             int                         src_line_size,
                             int                         dst_width,
                             int                         dst_height,
                             int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_x,
#else
                             const float*                pos_x,
#endif
                             const int*                  pos_x_0_suppress,
                             const int*                  pos_x_1_suppress,
                             const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                             const unsigned short int*   pos_y,
#else
                             const float*                pos_y,
#endif
                             const int*                  pos_y_0_suppress,
                             const int*                  pos_y_1_suppress,
                             unsigned char*              src_row_extract_buf,
                             unsigned int                alpha_flag)
{
    int width_aligned      = RESIZE_UC1_PROC_ALIGNED * (dst_width / RESIZE_UC1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_uc4_implement(src, 
                                   dst, 
                                   src_line_size, 
                                   width_aligned, 
                                   dst_height, 
                                   dst_line_size, 
                                   pos_x, 
                                   pos_x_0_suppress,
                                   pos_x_1_limit,
                                   pos_y,
                                   pos_y_0_suppress,
                                   pos_y_1_suppress,
                                   src_row_extract_buf,
                                   alpha_flag);
    }
#ifdef RESIZE_UC_USE_FIXED_PT
    for(i = 0 ; i < dst_height ; i ++)
    {
        unsigned int   v1            = pos_y[2 * i];
        unsigned int   v0            = pos_y[2 * i + 1];
        int            pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int            pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            unsigned int   res0        = 0;
            unsigned int   res1        = 0;
            unsigned int   res2        = 0;
            unsigned int   res3        = 0;
            unsigned int   u1          = pos_x[2 * j];
            unsigned int   u0          = pos_x[2 * j + 1];
            unsigned int   src_cur_0   = src[pos_y_0_start + 4 * pos_x_0_suppress[j]];
            unsigned int   src_cur_1   = src[pos_y_0_start + 4 * pos_x_1_suppress[j]];
            unsigned int   src_cur_2   = src[pos_y_1_start + 4 * pos_x_0_suppress[j]];
            unsigned int   src_cur_3   = src[pos_y_1_start + 4 * pos_x_1_suppress[j]];
            /*
            unsigned int   res         = (((((src_cur_0 * u1 +  src_cur_1 * u0) >> 4) * v1) >> 16) +
                                          ((((src_cur_2 * u1 +  src_cur_3 * u0) >> 4) * v0) >> 16)
                                          + 2) >> 2;
            */
            res0                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 4 * j] = res0 > 255 ? 255 : res0;

            src_cur_0                  = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_1                  = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 1];
            src_cur_2                  = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_3                  = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 1];
            res1                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 4 * j + 1] = res1 > 255 ? 255 : res1;
            src_cur_0                  = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_1                  = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 2];
            src_cur_2                  = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_3                  = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 2];
            res2                       = (src_cur_0 * u1 * v1 + 
                                          src_cur_1 * u0 * v1 + 
                                          src_cur_2 * u1 * v0 + 
                                          src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
            dst[i * dst_line_size + 4 * j + 2] = res2 > 255 ? 255 : res2;

            if(alpha_flag > 255)
            {
                src_cur_0                  = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 3];
                src_cur_1                  = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 3];
                src_cur_2                  = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 3];
                src_cur_3                  = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 3];
                res3                       = (src_cur_0 * u1 * v1 + 
                                            src_cur_1 * u0 * v1 + 
                                            src_cur_2 * u1 * v0 + 
                                            src_cur_3 * u0 * v0 + (1 << 21)) >> 22;
                dst[i * dst_line_size + 4 * j + 3] = res3 > 255 ? 255 : ((unsigned char)res3);
            }
            else
                dst[i * dst_line_size + 4 * j + 3] = (unsigned char)alpha_flag; 
        }
    }
#else
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + 4 * pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + 4 * pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + 4 * pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + 4 * pos_x_1_suppress[j]];
            float  res0, res1, res2, res3;
            res0               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            src_cur_0          = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_1          = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 1];
            src_cur_2          = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_3          = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 1];
            res1               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            src_cur_0          = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_1          = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 2];
            src_cur_2          = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_3          = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 2];
            res2               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            if(alpha_flag > 255)
            {
                src_cur_0      = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 3];
                src_cur_1      = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 3];
                src_cur_2      = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 3];
                src_cur_3      = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 3];
                res3           = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                    src_cur_1 * u * (1.0f - v) +
                                    src_cur_2 * (1.0f - u) * v +
                                    src_cur_3 * u * v);
            }
            dst[i * dst_line_size + 4 * j]     = res0 > 255 ? 255 : ((unsigned char)res0);
            dst[i * dst_line_size + 4 * j + 1] = res1 > 255 ? 255 : ((unsigned char)res1);
            dst[i * dst_line_size + 4 * j + 2] = res2 > 255 ? 255 : ((unsigned char)res2);
            if(alpha_flag <= 255)
                dst[i * dst_line_size + 4 * j + 3] = (unsigned char)alpha_flag;
            else
                dst[i * dst_line_size + 4 * j + 3] = res3 > 255 ? 255 : ((unsigned char)res3);
        }
    }
#endif
}

void  dlcv_resize_image_uc(const unsigned char*  src,
                           unsigned char*        dst,
                           int                   src_width,
                           int                   src_height,
                           int                   src_line_size,
                           int                   dst_width,
                           int                   dst_height,
                           int                   dst_line_size,
                           int                   channel_cnt,
                           unsigned int          alpha_flag)
{
    int              i                       = 0;
    int              j                       = 0;
    int              pos_x_1_valid[2]        = { dst_width - 1, -1 } ;
    float            src2dst_ratio[2]        = { 0.0f, 0.0f };
    int              pos_x_buf_len           = dst_width * (sizeof(int));
    int              pos_y_buf_len           = dst_height * (sizeof(int));
    int              pos_x_buf_len_aligned32 = (((pos_x_buf_len + 31) >> 5) << 5) ;
    int              pos_y_buf_len_aligned32 = (((pos_y_buf_len + 31) >> 5) << 5) ;
    int              src_row_extract_buf_len = 4 * (3 == channel_cnt ? 4 : channel_cnt) * ((((dst_width * 4) + 31) >> 5) << 5);
    int              work_buf_len_aligned    = 3 * (pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32) + src_row_extract_buf_len;
    unsigned char*   work_buf                = (unsigned char*)dlcv_malloc_mem(work_buf_len_aligned);
    float*           pos_x_f                 = (float*)(work_buf);
    int*             pos_x_0                 = (int*)(work_buf + pos_x_buf_len_aligned32);
    int*             pos_x_1                 = (int*)(work_buf + 2 * pos_x_buf_len_aligned32);
    float*           pos_y_f                 = (float*)(work_buf + 3 * pos_x_buf_len_aligned32);
    int*             pos_y_0                 = (int*)(work_buf + 3 * pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32);
    int*             pos_y_1                 = (int*)(work_buf + 3 * pos_x_buf_len_aligned32 + 2 * pos_y_buf_len_aligned32);
    unsigned char*   src_row_extract_buf     = (unsigned char*)(work_buf + 3 * (pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32));
    memset(work_buf, 0, work_buf_len_aligned);

    src2dst_ratio[0] = (float)src_width / ((float)dst_width);
    src2dst_ratio[1] = (float)src_height / ((float)dst_height);

#ifdef RESIZE_UC_USE_FIXED_PT
    caculate_pos_fixpt_opencv(src_width,  dst_width,  (unsigned short int*)pos_x_f, pos_x_0, pos_x_1, src2dst_ratio[0]);
    caculate_pos_fixpt_opencv(src_height, dst_height, (unsigned short int*)pos_y_f, pos_y_0, pos_y_1, src2dst_ratio[1]);
#else
    caculate_pos_float_opencv(src_width,  dst_width,  pos_x_f, pos_x_0, pos_x_1, src2dst_ratio[0]);
    caculate_pos_float_opencv(src_height, dst_height, pos_y_f, pos_y_0, pos_y_1, src2dst_ratio[1]);
#endif
    for(i = 0 ; i < dst_width ; i ++)
    {
        if(0 != pos_x_1[i])
        {
            pos_x_1_valid[0] = i;
            break;
        }
    }

    for(i = dst_width - 1 ; i >= 0 ; i --)
    {
        if(1 == abs(pos_x_1[i] - pos_x_0[i]))
        {
            pos_x_1_valid[1] = i;
            break;
        }
    }

    if(1 == channel_cnt)
    {
        resize_image_uc1(src, 
                         dst,
                         src_line_size, 
                         dst_width, 
                         dst_height, 
                         dst_line_size, 
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_x_f,
#else
                         pos_x_f,
#endif
                         pos_x_0,
                         pos_x_1,
                         pos_x_1_valid,
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_y_f,
#else
                         pos_y_f,
#endif
                         pos_y_0,
                         pos_y_1,
                         src_row_extract_buf);
        
    }
    else if(3 == channel_cnt)
    {
        resize_image_uc3(src, 
                         dst,
                         src_line_size, 
                         dst_width, 
                         dst_height, 
                         dst_line_size, 
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_x_f,
#else
                         pos_x_f,
#endif
                         pos_x_0,
                         pos_x_1,
                         pos_x_1_valid,
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_y_f,
#else
                         pos_y_f,
#endif
                         pos_y_0,
                         pos_y_1,
                         src_row_extract_buf);
    }
    else
    {
        resize_image_uc4(src, 
                         dst,
                         src_line_size, 
                         dst_width, 
                         dst_height, 
                         dst_line_size, 
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_x_f,
#else
                         pos_x_f,
#endif
                         pos_x_0,
                         pos_x_1,
                         pos_x_1_valid,
#ifdef RESIZE_UC_USE_FIXED_PT
                         (const unsigned short int*)pos_y_f,
#else
                         pos_y_f,
#endif
                         pos_y_0,
                         pos_y_1,
                         src_row_extract_buf,
                         alpha_flag);
    }
    
    dlcv_free_mem(work_buf);
}

#if 0
static void resize_image_f1(const float*        src,
                            float*              dst,
                            int                 src_line_size,
                            int                 dst_width,
                            int                 dst_height,
                            int                 dst_line_size,
                            const float*        pos_x,
                            const int*          pos_x_0_suppress,
                            const int*          pos_x_1_suppress,
                            const int*          pos_x_1_limit,
                            const float*        pos_y,
                            const int*          pos_y_0_suppress,
                            const int*          pos_y_1_suppress,
                            float*              src_row_extract_buf)
{
    int width_aligned      = RESIZE_F1_PROC_ALIGNED * (dst_width / RESIZE_F1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_f1_implement(src, 
                                  dst, 
                                  src_line_size, 
                                  width_aligned, 
                                  dst_height, 
                                  dst_line_size, 
                                  pos_x, 
                                  pos_x_0_suppress,
                                  pos_x_1_limit,
                                  pos_y,
                                  pos_y_0_suppress,
                                  pos_y_1_suppress,
                                  src_row_extract_buf);
    }
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + pos_x_1_suppress[j]];
            float        res   = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + j] = res;
        }
    }
}

static void resize_image_f3(const float*        src,
                            float*              dst,
                            int                 src_line_size,
                            int                 dst_width,
                            int                 dst_height,
                            int                 dst_line_size,
                            const float*        pos_x,
                            const int*          pos_x_0_suppress,
                            const int*          pos_x_1_suppress,
                            const int*          pos_x_1_limit,
                            const float*        pos_y,
                            const int*          pos_y_0_suppress,
                            const int*          pos_y_1_suppress,
                            float*              src_row_extract_buf)
{
#if 1   
    int width_aligned      = RESIZE_F1_PROC_ALIGNED * (dst_width / RESIZE_F1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_f3_implement(src, 
                                  dst, 
                                  src_line_size, 
                                  width_aligned, 
                                  dst_height, 
                                  dst_line_size, 
                                  pos_x, 
                                  pos_x_0_suppress,
                                  pos_x_1_limit,
                                  pos_y,
                                  pos_y_0_suppress,
                                  pos_y_1_suppress,
                                  src_row_extract_buf);
    }
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  res0, res1, res2;
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j]];
            res0               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 3 * j] = res0;
            src_cur_0          = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_1          = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 1];
            src_cur_2          = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 1];
            src_cur_3          = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 1];
            res1               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 3 * j + 1] = res1;

            src_cur_0          = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_1          = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 2];
            src_cur_2          = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 2];
            src_cur_3          = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 2];
            res2               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 3 * j + 2] = res2;
        }
    }
#else
    for(int i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(int j = 0 ; j < dst_width ; j ++)
        {
            float  u           = pos_x[j];
            {
                float  src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j]];
                float  src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j]];
                float  src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j]];
                float  src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j]];
                float  res0 = src_cur_0 * (1.0f - u) + src_cur_1 * u;
                float  res1 = src_cur_2 * (1.0f - u) + src_cur_3 * u;
                res0        = res0 * (1.0f - v) + res1 * v;
                dst[i * dst_line_size + 3 * j] = res0;
            }

            {
                float  src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 1];
                float  src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 1];
                float  src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 1];
                float  src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 1];
                float  res0 = src_cur_0 * (1.0f - u) + src_cur_1 * u;
                float  res1 = src_cur_2 * (1.0f - u) + src_cur_3 * u;
                res0        = res0 * (1.0f - v) + res1 * v;
                dst[i * dst_line_size + 3 * j + 1] = res0;
            }
            
            {
                float  src_cur_0   = src[pos_y_0_start + 3 * pos_x_0_suppress[j] + 2];
                float  src_cur_1   = src[pos_y_0_start + 3 * pos_x_1_suppress[j] + 2];
                float  src_cur_2   = src[pos_y_1_start + 3 * pos_x_0_suppress[j] + 2];
                float  src_cur_3   = src[pos_y_1_start + 3 * pos_x_1_suppress[j] + 2];
                float  res0 = src_cur_0 * (1.0f - u) + src_cur_1 * u;
                float  res1 = src_cur_2 * (1.0f - u) + src_cur_3 * u;
                res0        = res0 * (1.0f - v) + res1 * v;
                dst[i * dst_line_size + 3 * j + 2] = res0;
            }
        }
    }
#endif
}

static void resize_image_f4(const float*        src,
                            float*              dst,
                            int                 src_line_size,
                            int                 dst_width,
                            int                 dst_height,
                            int                 dst_line_size,
                            const float*        pos_x,
                            const int*          pos_x_0_suppress,
                            const int*          pos_x_1_suppress,
                            const int*          pos_x_1_limit,
                            const float*        pos_y,
                            const int*          pos_y_0_suppress,
                            const int*          pos_y_1_suppress,
                            float*              src_row_extract_buf)
{ 
    int width_aligned      = RESIZE_F1_PROC_ALIGNED * (dst_width / RESIZE_F1_PROC_ALIGNED);
    int i                  = 0;
    int j                  = 0;
    if(width_aligned > 0)
    {
        resize_image_f4_implement(src, 
                                  dst, 
                                  src_line_size, 
                                  width_aligned, 
                                  dst_height, 
                                  dst_line_size, 
                                  pos_x, 
                                  pos_x_0_suppress,
                                  pos_x_1_limit,
                                  pos_y,
                                  pos_y_0_suppress,
                                  pos_y_1_suppress,
                                  src_row_extract_buf);
    }
    for(i = 0 ; i < dst_height ; i ++)
    {
        int    pos_y_0_start = pos_y_0_suppress[i] * src_line_size;
        int    pos_y_1_start = pos_y_1_suppress[i] * src_line_size;
        float  v             = pos_y[i];
        for(j = width_aligned ; j < dst_width ; j ++)
        {
            float  res0, res1, res2, res3;
            float  u           = pos_x[j];
            float  src_cur_0   = src[pos_y_0_start + 4 * pos_x_0_suppress[j]];
            float  src_cur_1   = src[pos_y_0_start + 4 * pos_x_1_suppress[j]];
            float  src_cur_2   = src[pos_y_1_start + 4 * pos_x_0_suppress[j]];
            float  src_cur_3   = src[pos_y_1_start + 4 * pos_x_1_suppress[j]];
            res0               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 4 * j] = res0;
            src_cur_0          = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_1          = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 1];
            src_cur_2          = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 1];
            src_cur_3          = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 1];
            res1               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 4 * j + 1] = res1;

            src_cur_0          = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_1          = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 2];
            src_cur_2          = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 2];
            src_cur_3          = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 2];
            res2               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 4 * j + 2] = res2;

            src_cur_0          = src[pos_y_0_start + 4 * pos_x_0_suppress[j] + 3];
            src_cur_1          = src[pos_y_0_start + 4 * pos_x_1_suppress[j] + 3];
            src_cur_2          = src[pos_y_1_start + 4 * pos_x_0_suppress[j] + 3];
            src_cur_3          = src[pos_y_1_start + 4 * pos_x_1_suppress[j] + 3];
            res3               = (src_cur_0 * (1.0f - u) * (1.0f - v) + 
                                  src_cur_1 * u * (1.0f - v) +
                                  src_cur_2 * (1.0f - u) * v +
                                  src_cur_3 * u * v);
            dst[i * dst_line_size + 4 * j + 3] = res3;
        }
    }
}

void  dlcv_resize_image_float(const float*       src,
                              float*             dst,
                              int                src_width,
                              int                src_height,
                              int                src_line_size,
                              int                dst_width,
                              int                dst_height,
                              int                dst_line_size,
                              int                channel_cnt,
                              unsigned int       alpha_as_zero_flag)
{
    int              i                       = 0;
    int              j                       = 0;
    int              pos_x_1_valid[2]        = { dst_width - 1, -1 } ;
    float            src2dst_ratio[2]        = { 0.0f, 0.0f };
    int              pos_x_buf_len           = dst_width * (sizeof(int));
    int              pos_y_buf_len           = dst_height * (sizeof(int));
    int              pos_x_buf_len_aligned32 = (((pos_x_buf_len + 31) >> 5) << 5) ;
    int              pos_y_buf_len_aligned32 = (((pos_y_buf_len + 31) >> 5) << 5) ;
    int              src_row_extract_buf_len = 4 * (3 == channel_cnt ? 4 : channel_cnt) * ((((dst_width * sizeof(float) * 4) + 31) >> 5) << 5);
    int              work_buf_len_aligned    = 3 * (pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32) + src_row_extract_buf_len;
    unsigned char*   work_buf                = (unsigned char*)dlcv_malloc_mem(work_buf_len_aligned);
    float*           pos_x_f                 = (float*)(work_buf);
    int*             pos_x_0                 = (int*)(work_buf + pos_x_buf_len_aligned32);
    int*             pos_x_1                 = (int*)(work_buf + 2 * pos_x_buf_len_aligned32);
    float*           pos_y_f                 = (float*)(work_buf + 3 * pos_x_buf_len_aligned32);
    int*             pos_y_0                 = (int*)(work_buf + 3 * pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32);
    int*             pos_y_1                 = (int*)(work_buf + 3 * pos_x_buf_len_aligned32 + 2 * pos_y_buf_len_aligned32);
    unsigned char*   src_row_extract_buf     = (unsigned char*)(work_buf + 3 * (pos_x_buf_len_aligned32 + pos_y_buf_len_aligned32));
    memset(work_buf, 0, work_buf_len_aligned);

    src2dst_ratio[0] = (float)src_width / ((float)dst_width);
    src2dst_ratio[1] = (float)src_height / ((float)dst_height);

    caculate_pos_float_opencv(src_width,  dst_width,  pos_x_f, pos_x_0, pos_x_1, src2dst_ratio[0]);
    caculate_pos_float_opencv(src_height, dst_height, pos_y_f, pos_y_0, pos_y_1, src2dst_ratio[1]);

    for(i = 0 ; i < dst_width ; i ++)
    {
        if(0 != pos_x_1[i])
        {
            pos_x_1_valid[0] = i;
            break;
        }
    }

    for(i = dst_width - 1 ; i >= 0 ; i --)
    {
        if(1 == abs(pos_x_1[i] - pos_x_0[i]))
        {
            pos_x_1_valid[1] = i;
            break;
        }
    }

    if(1 == channel_cnt)
    {
        resize_image_f1(src, 
                        dst,
                        src_line_size, 
                        dst_width, 
                        dst_height, 
                        dst_line_size, 
                        pos_x_f,
                        pos_x_0,
                        pos_x_1,
                        pos_x_1_valid,
                        pos_y_f,
                        pos_y_0,
                        pos_y_1,
                        (float*)src_row_extract_buf);
        
    }
    else if(3 == channel_cnt)
    {
        resize_image_f3(src, 
                        dst,
                        src_line_size, 
                        dst_width, 
                        dst_height, 
                        dst_line_size, 
                        pos_x_f,
                        pos_x_0,
                        pos_x_1,
                        pos_x_1_valid,
                        pos_y_f,
                        pos_y_0,
                        pos_y_1,
                        (float*)src_row_extract_buf);
    }
    else
    {
        resize_image_f4(src, 
                        dst,
                        src_line_size, 
                        dst_width, 
                        dst_height, 
                        dst_line_size, 
                        pos_x_f,
                        pos_x_0,
                        pos_x_1,
                        pos_x_1_valid,
                        pos_y_f,
                        pos_y_0,
                        pos_y_1,
                        (float*)src_row_extract_buf);
    }
    
    dlcv_free_mem(work_buf);
}
#endif