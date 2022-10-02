#include "../dlcv_proc_opt.h"
#include "./normalize_image.h"

void  dlcv_normalize_image_uc1f1(const unsigned char*   src,
                                 float*                 dst,
                                 int                    width,
                                 int                    height,
                                 int                    src_line_size,
                                 int                    dst_line_size,
                                 const float*           mean,
                                 const float*           std)
{
    float  mean_val         = mean[0];
    float  inv_std          = 1.0f / std[0];
    int    i                = 0;
    int    j                = 0;
    int    linesize_aligned = UC1F1_PROC_ALIGNED * (width / UC1F1_PROC_ALIGNED);
    int    width_aligned    = linesize_aligned;
    if(width_aligned > 0)
        normalize_image_uc1f1_implement(src, dst, width_aligned, height, src_line_size, dst_line_size, mean_val, inv_std);
    if (linesize_aligned < src_line_size)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < width; j ++)
            {
                dst[dst_line_size * i + j] = (((float)src[src_line_size * i + j]) - mean_val) * inv_std;
            }
        }
    }
}

void  dlcv_normalize_image_f1f1(const float*   src,
                                float*         dst,
                                int            width,
                                int            height,
                                int            src_line_size,
                                int            dst_line_size,
                                const float*   mean,
                                const float*   std)
{
    float  mean_val         = mean[0];
    float  inv_std          = 1.0f / std[0];
    int    i                = 0;
    int    j                = 0;
    int    linesize_aligned = F1F1_PROC_ALIGNED * (width / F1F1_PROC_ALIGNED);
    int    width_aligned    = linesize_aligned;
    if(width_aligned > 0)
        normalize_image_f1f1_implement(src, dst, width_aligned, height, src_line_size, dst_line_size, mean_val, inv_std);
    if (linesize_aligned < src_line_size)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < width; j ++)
            {
                dst[dst_line_size * i + j] = (src[src_line_size * i + j] - mean_val) * inv_std;
            }
        }
    }
}

void  dlcv_normalize_image_uc3f3(const unsigned char*   src,
                                 float*                 dst,
                                 int                    width,
                                 int                    height,
                                 int                    src_line_size,
                                 int                    dst_line_size,
                                 const float*           mean,
                                 const float*           std)
{
    float   mean_val[24]  = { 0.0f };
    float   std_val[24]   = { 0.0f };
    int     i             = 0;
    int     j             = 0;
    for (i = 0; i < 24; i += 3)
    {
        mean_val[i]     = mean[0];
        mean_val[i + 1] = mean[1];
        mean_val[i + 2] = mean[2];
    }

    for (i = 0; i < 24; i += 3)
    {
        std_val[i]     = 1.0f / std[0];
        std_val[i + 1] = 1.0f / std[1];
        std_val[i + 2] = 1.0f / std[2];
    }
    int                  linesize         = 3 * width;
    int                  linesize_aligned = ((3 * width) / UC3F3_PROC_ALIGNED) * UC3F3_PROC_ALIGNED;
    if(linesize_aligned > 0)
        normalize_image_uc3f3_implement(src, dst, linesize_aligned/3, height, src_line_size, dst_line_size, mean_val, std_val);

    if (linesize_aligned < linesize)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < linesize; j += 3)
            {
                dst[dst_line_size * i + j]     = ((float)(src[src_line_size * i + j])     - mean_val[0]) * std_val[0];
                dst[dst_line_size * i + j + 1] = ((float)(src[src_line_size * i + j + 1]) - mean_val[1]) * std_val[1];
                dst[dst_line_size * i + j + 2] = ((float)(src[src_line_size * i + j + 2]) - mean_val[2]) * std_val[2];
            }
        }
    }
}

void  dlcv_normalize_image_f3f3(const float*   src, 
                               float*         dst, 
                               int            width, 
                               int            height, 
                               int            src_line_size, 
                               int            dst_line_size, 
                               const float*   mean, 
                               const float*   std)
{
    float   mean_val[24]  = { 0.0f };
    float   std_val[24]   = { 0.0f };
    int     i             = 0;
    int     j             = 0;
    for (i = 0; i < 24; i += 3)
    {
        mean_val[i]     = mean[0];
        mean_val[i + 1] = mean[1];
        mean_val[i + 2] = mean[2];
    }

    for (i = 0; i < 24; i += 3)
    {
        std_val[i]     = 1.0f / std[0];
        std_val[i + 1] = 1.0f / std[1];
        std_val[i + 2] = 1.0f / std[2];
    }
    int                  linesize         = 3 * width;
    int                  linesize_aligned = ((3 * width) / F3F3_PROC_ALIGNED) * F3F3_PROC_ALIGNED;
    if(linesize_aligned > 0)
        normalize_image_f3f3_implement(src, dst, linesize_aligned/3, height, src_line_size, dst_line_size, mean_val, std_val);

    if (linesize_aligned < linesize)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < linesize; j += 3)
            {
                dst[dst_line_size * i + j]     = (src[src_line_size * i + j]     - mean_val[0]) * std_val[0];
                dst[dst_line_size * i + j + 1] = (src[src_line_size * i + j + 1] - mean_val[1]) * std_val[1];
                dst[dst_line_size * i + j + 2] = (src[src_line_size * i + j + 2] - mean_val[2]) * std_val[2];
            }
        }
    }
}

void  dlcv_normalize_image_uc4f4(const unsigned char*   src,
                                 float*                 dst,
                                 int                    width,
                                 int                    height,
                                 int                    src_line_size,
                                 int                    dst_line_size,
                                 const float*           mean,
                                 const float*           std)
{
    float   mean_val[8]  = {     
        mean[0],     mean[1],     mean[2],     mean[3],
        mean[0],     mean[1],     mean[2],     mean[3] 
    };
    float   std_val[8]   = { 
        1.0f/std[0], 1.0f/std[1], 1.0f/std[2], 1.0f/std[3], 
        1.0f/std[0], 1.0f/std[1], 1.0f/std[2], 1.0f/std[3]
    };
    int     i            = 0;
    int     j            = 0;

    int     linesize         = 4 * width;
    int     linesize_aligned = ((4 * width) / UC4F4_PROC_ALIGNED) * UC4F4_PROC_ALIGNED;
    if(linesize_aligned > 0)
        normalize_image_uc4f4_implement(src, dst, linesize_aligned/4, height, src_line_size, dst_line_size, mean_val, std_val);

    if (linesize_aligned < linesize)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < linesize; j += 4)
            {
                dst[dst_line_size * i + j]     = ((float)(src[src_line_size * i + j])     - mean_val[0]) * std_val[0];
                dst[dst_line_size * i + j + 1] = ((float)(src[src_line_size * i + j + 1]) - mean_val[1]) * std_val[1];
                dst[dst_line_size * i + j + 2] = ((float)(src[src_line_size * i + j + 2]) - mean_val[2]) * std_val[2];
                dst[dst_line_size * i + j + 3] = ((float)(src[src_line_size * i + j + 3]) - mean_val[3]) * std_val[3];
            }
        }
    }
}

void  dlcv_normalize_image_f4f4(const float*   src,
                                float*         dst,
                                int            width,
                                int            height,
                                int            src_line_size,
                                int            dst_line_size,
                                const float*   mean,
                                const float*   std)
{
    float   mean_val[8]  = {     
        mean[0],     mean[1],     mean[2],     mean[3],
        mean[0],     mean[1],     mean[2],     mean[3] 
    };
    float   std_val[8]   = { 
        1.0f/std[0], 1.0f/std[1], 1.0f/std[2], 1.0f/std[3], 
        1.0f/std[0], 1.0f/std[1], 1.0f/std[2], 1.0f/std[3]
    };
    int     i            = 0;
    int     j            = 0;

    int     linesize         = 4 * width;
    int     linesize_aligned = ((4 * width) / F4F4_PROC_ALIGNED) * F4F4_PROC_ALIGNED;
    if(linesize_aligned > 0)
        normalize_image_f4f4_implement(src, dst, linesize_aligned/4, height, src_line_size, dst_line_size, mean_val, std_val);

    if (linesize_aligned < linesize)
    {
        for (i = 0; i < height; i++)
        {
            for (j = linesize_aligned; j < linesize; j += 4)
            {
                dst[dst_line_size * i + j]     = (src[src_line_size * i + j]     - mean_val[0]) * std_val[0];
                dst[dst_line_size * i + j + 1] = (src[src_line_size * i + j + 1] - mean_val[1]) * std_val[1];
                dst[dst_line_size * i + j + 2] = (src[src_line_size * i + j + 2] - mean_val[2]) * std_val[2];
                dst[dst_line_size * i + j + 3] = (src[src_line_size * i + j + 3] - mean_val[3]) * std_val[3];
            }
        }
    }
}