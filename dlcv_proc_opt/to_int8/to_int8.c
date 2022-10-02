#include "to_int8.h"

static void proc_tail(float const*      src,
                      unsigned char*    dst,
                      int               data_length,
                      int               data_aligned,
                      float             scale,
                      float             bias)
{
    int i = 0;
    for(i = data_aligned ; i < data_length ; i ++)
    {
        float a = src[i];
        a       = a * scale + bias;
        int g   = (int)(a >= 0.0f ? a + 0.5f : 0.0f);
        dst[i]  = ((unsigned char)(g > 255 ? 255 : g));
    }
}

static void proc_tail_2d(float const*      src,
                         unsigned char*    dst,
                         int               src_width,
                         int               width_aligned,
                         int               src_height,
                         int               src_line_size,
                         int               dst_line_size,
                         float             scale,
                         float             bias)
{
    int i = 0, j = 0;
    for(i = 0 ; i < src_height ; i ++)
    {
        for(j = width_aligned ; j < src_width ; j ++)
        {
            float a = src[i * src_line_size + j];
            a       = a * scale + bias;
            int g   = (int)(a >= 0.0f ? a + 0.5f : 0.0f);
            dst[i * dst_line_size + j] = ((unsigned char)(g > 255 ? 255 : g));
        }
    }
}

void dlcv_image_to_int8_f1uc1(float const*      src,
                              unsigned char*    dst,
                              int               src_width,
                              int               src_height,
                              int               src_line_size,
                              int               dst_line_size,
                              float             step,
                              float             zero)
{
    float  scale         = 1.0f / step;
    if(src_width == src_line_size && src_width == dst_line_size)
    {
        int   data_length    = src_width * src_height;
        int   length_aligned = ((data_length >> 5) << 5);
        if(length_aligned > 0)
            data_to_int8_implement(src, dst, length_aligned, scale, zero);
            
        if(length_aligned < data_length)
            proc_tail(src, dst, data_length, length_aligned, scale, zero);
    }
    else
    {
        int    width_aligned = ((src_width >> 4) << 4);
        if(width_aligned > 0)
            image_to_int8_f1uc1_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, scale, zero);
        if(width_aligned < src_width)
            proc_tail_2d(src, dst, src_width, width_aligned, src_height, src_line_size, dst_line_size, scale, zero);
    }
}

void dlcv_image_to_int8_f3uc3(float const*      src,
                              unsigned char*    dst,
                              int               src_width,
                              int               src_height,
                              int               src_line_size,
                              int               dst_line_size,
                              float             step,
                              float             zero)
{
    float  scale         = 1.0f / step;
    int    src_width3    = 3 * src_width;
    if(src_width3 == src_line_size && src_width3 == dst_line_size)
    {
        int   data_length    = src_width3 * src_height;
        int   length_aligned = ((data_length >> 5) << 5);
        if(length_aligned > 0)
            data_to_int8_implement(src, dst, length_aligned, scale, zero);

        if(length_aligned < data_length)
            proc_tail(src, dst, data_length, length_aligned, scale, zero);
    }
    else
    {
        int    width_aligned = ((src_width3 >> 4) << 4);
        if(width_aligned > 0)
            image_to_int8_f1uc1_implement(src, dst, width_aligned, src_height, src_line_size, dst_line_size, scale, zero);
        if(width_aligned < src_width3)
            proc_tail_2d(src, dst, src_width3, width_aligned, src_height, src_line_size, dst_line_size, scale, zero);
    }
}