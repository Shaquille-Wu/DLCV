#include "to_gray.h"

static const unsigned int  kCvtCoefBits = 15;
static const unsigned int  kCvtCoefBase = (1 << 15);
static const float         kCvtCoef[]   = {
    0.114f,       //B
    0.587f,       //G
    0.299f        //R
};

static void image_togray_uc3(const unsigned char*        src,
                             unsigned char*              dst,
                             int                         src_width,
                             int                         src_height,
                             int                         src_line_size,
                             int                         dst_line_size,
                             unsigned short int const*   cvt_coef)
{
    int  width_aligned = ((src_width >> 4) << 4);

    if(width_aligned > 0)
        image_togray_uc3_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, cvt_coef);

    if(width_aligned < src_width)
    {
        int i = 0, j = 0;
        unsigned int  c0 = cvt_coef[0];
        unsigned int  c1 = cvt_coef[1];
        unsigned int  c2 = cvt_coef[2];
        for(i = 0 ; i < src_height ; i ++)
        {
            for(j = width_aligned ; j < src_width ; j ++)
            {
                unsigned int a = src[i * src_line_size + 3 * j];
                unsigned int b = src[i * src_line_size + 3 * j + 1];
                unsigned int c = src[i * src_line_size + 3 * j + 2];
                unsigned int g = (a * c0 + b * c1 + c * c2 + 16384) >> kCvtCoefBits;
                dst[i * dst_line_size + j] = ((unsigned char)g);
            }
        }
    }
}

static void image_togray_uc4(const unsigned char*        src,
                             unsigned char*              dst,
                             int                         src_width,
                             int                         src_height,
                             int                         src_line_size,
                             int                         dst_line_size,
                             unsigned short int const*   cvt_coef)
{
    int  width_aligned = ((src_width >> 4) << 4);

    if(width_aligned > 0)
        image_togray_uc4_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, cvt_coef);

    if(width_aligned < src_width)
    {
        int i = 0, j = 0;
        unsigned int  c0 = cvt_coef[0];
        unsigned int  c1 = cvt_coef[1];
        unsigned int  c2 = cvt_coef[2];
        for(i = 0 ; i < src_height ; i ++)
        {
            for(j = width_aligned ; j < src_width ; j ++)
            {
                unsigned int a = src[i * src_line_size + 4 * j];
                unsigned int b = src[i * src_line_size + 4 * j + 1];
                unsigned int c = src[i * src_line_size + 4 * j + 2];
                unsigned int g = (a * c0 + b * c1 + c * c2 + 16384) >> kCvtCoefBits;
                dst[i * dst_line_size + j] = ((unsigned char)g);
            }
        }
    }
}

void  dlcv_image_togray_uc(const unsigned char*   src,
                           unsigned char*         dst,
                           int                    src_width,
                           int                    src_height,
                           int                    src_line_size,
                           int                    dst_line_size,
                           int                    src_channel,   //just 3 or 4
                           int                    bgr_or_rgb     //0 is bgr, 1 is rgb
                           )
{
    unsigned short int  cvt_coef[4]  = { 0, 0, 0, 0 };
    int                 i            = 0;
    if(3 != src_channel && 4 != src_channel)   return;

    if(0 == bgr_or_rgb)
    {
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
        cvt_coef[0] = 3736;
        cvt_coef[1] = 19234;
#else
        cvt_coef[0] = 3735;
        cvt_coef[1] = 19235;
#endif
        cvt_coef[2] = 9798;
    }
    else
    {
        cvt_coef[0] = 9798;
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
        cvt_coef[1] = 19234;
        cvt_coef[2] = 3736;
#else
        cvt_coef[1] = 19235;
        cvt_coef[2] = 3735;
#endif
    }

    if(3 == src_channel)
        image_togray_uc3(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
    else
        image_togray_uc4(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}

static void image_togray_f3(float const*   src,
                            float*         dst,
                            int            src_width,
                            int            src_height,
                            int            src_line_size,
                            int            dst_line_size,
                            float const*   cvt_coef)
{
    int  width_aligned = ((src_width >> 3) << 3);

    if(width_aligned > 0)
        image_togray_f3_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, cvt_coef);

    if(width_aligned < src_width)
    {
        int i = 0, j = 0;
        float  c0 = cvt_coef[0];
        float  c1 = cvt_coef[1];
        float  c2 = cvt_coef[2];
        for(i = 0 ; i < src_height ; i ++)
        {
            for(j = width_aligned ; j < src_width ; j ++)
            {
                float a = src[i * src_line_size + 3 * j];
                float b = src[i * src_line_size + 3 * j + 1];
                float c = src[i * src_line_size + 3 * j + 2];
                float g = (a * c0 + b * c1 + c * c2);
                dst[i * dst_line_size + j] = g;
            }
        }
    }
}

static void image_togray_f4(float const*   src,
                            float*         dst,
                            int            src_width,
                            int            src_height,
                            int            src_line_size,
                            int            dst_line_size,
                            float const*   cvt_coef)
{
    int  width_aligned = ((src_width >> 3) << 3);

    if(width_aligned > 0)
        image_togray_f4_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, cvt_coef);

    if(width_aligned < src_width)
    {
        int i = 0, j = 0;
        float  c0 = cvt_coef[0];
        float  c1 = cvt_coef[1];
        float  c2 = cvt_coef[2];
        for(i = 0 ; i < src_height ; i ++)
        {
            for(j = width_aligned ; j < src_width ; j ++)
            {
                float a = src[i * src_line_size + 4 * j];
                float b = src[i * src_line_size + 4 * j + 1];
                float c = src[i * src_line_size + 4 * j + 2];
                float g = (a * c0 + b * c1 + c * c2);
                dst[i * dst_line_size + j] = g;
            }
        }
    }
}

void  dlcv_image_togray_f(const float*           src,
                          float*                 dst,
                          int                    src_width,
                          int                    src_height,
                          int                    src_line_size,
                          int                    dst_line_size,
                          int                    src_channel,   //just 3 or 4
                          int                    bgr_or_rgb     //0 is bgr, 1 is rgb
                          )
{
    float  cvt_coef[4]  = { 0, 0, 0, 0 };
    int    i            = 0;
    if(3 != src_channel && 4 != src_channel)   return;

    if(0 == bgr_or_rgb)
    {
        cvt_coef[0] = kCvtCoef[0];
        cvt_coef[1] = kCvtCoef[1];
        cvt_coef[2] = kCvtCoef[2];
    }
    else
    {
        cvt_coef[0] = kCvtCoef[2];
        cvt_coef[1] = kCvtCoef[1];
        cvt_coef[2] = kCvtCoef[0];
    }

    if(3 == src_channel)
        image_togray_f3(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
    else
        image_togray_f4(src, dst, src_width, src_height, src_line_size, dst_line_size, cvt_coef);
}