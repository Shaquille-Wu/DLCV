#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../dlcv_proc_opt.h"
#include "./super_point.h"
#include "../common/memory_op.h"

static int   super_point_extract_key_points_8x(float const*        prob,
                                               int                 width,
                                               int                 height,
                                               int                 channel,
                                               float               threshold,
                                               int                 bkg_at_last,
                                               unsigned int*       point_flag,
                                               unsigned int*       point_list)
{
    float const*  cur_point_prob      = prob;
    unsigned int  i = 0, j = 0, k = 0;
    int           cur_point_valid_cnt = 0;
    int           total_valid_cnt     = 0;
    int           channel_class       = channel;
    unsigned int  idx_in_blk          = 0;
    unsigned int* cur_pt_list         = point_list;
    if(0 != bkg_at_last)
        channel_class = channel_class - 1;

    for(i = 0 ; i < height ; i ++)
    {
        for(j = 0 ; j < width; j ++)
        {
            cur_point_valid_cnt = super_point_extract_flag_implement(cur_point_prob, 
                                                                     channel_class, 
                                                                     (int*)point_flag, 
                                                                     threshold);
            for(k = 0 ; k < cur_point_valid_cnt ; k ++)
            {
                unsigned int x =  0 ;
                unsigned int y =  0 ;
                idx_in_blk     = point_flag[k];
                y                                = (i << 3) + (idx_in_blk >> 3);
                x                                = (j << 3) + (idx_in_blk & 7);
                cur_pt_list[2 * k]               = (y << 16) | x;
                ((float*)cur_pt_list)[2 * k + 1] = cur_point_prob[idx_in_blk];
            }
            cur_pt_list     += (2 * cur_point_valid_cnt);
            total_valid_cnt += cur_point_valid_cnt;
            cur_point_prob  += channel;
        }
    }
    return total_valid_cnt;
}

static int   super_point_extract_key_points(float const*        prob,
                                            int                 width,
                                            int                 height,
                                            int                 channel,
                                            int                 upsample_scale,
                                            float               threshold,
                                            int                 bkg_at_last,
                                            unsigned int*       point_flag,
                                            unsigned int*       point_list)
{
    float const*  cur_point_prob      = prob;
    unsigned int  i = 0, j = 0, k = 0;
    int           cur_point_valid_cnt = 0;
    int           total_valid_cnt     = 0;
    int           channel_class       = channel;
    unsigned int  idx_in_blk          = 0;
    unsigned int* cur_pt_list         = point_list;
    if(0 != bkg_at_last)
        channel_class = channel_class - 1;
    for(i = 0 ; i < height ; i ++)
    {
        for(j = 0 ; j < width; j ++)
        {
            cur_point_valid_cnt = super_point_extract_flag_implement(cur_point_prob, 
                                                                     channel_class, 
                                                                     (int*)point_flag, 
                                                                     threshold);
            for(k = 0 ; k < cur_point_valid_cnt ; k ++)
            {
                unsigned int x =  0 ;
                unsigned int y =  0 ;
                idx_in_blk                       = point_flag[k];
                y                                = i * ((unsigned int)upsample_scale) + (idx_in_blk / ((unsigned int)upsample_scale));
                x                                = j * ((unsigned int)upsample_scale) + (idx_in_blk % ((unsigned int)upsample_scale));
                cur_pt_list[2 * k]               = (y << 16) | x;
                ((float*)cur_pt_list)[2 * k + 1] = cur_point_prob[idx_in_blk];
            }
            cur_pt_list     += 2 * (cur_point_valid_cnt);
            total_valid_cnt += cur_point_valid_cnt;
            cur_point_prob  += channel;
        }
    }

    return total_valid_cnt;
}

static int   super_point_extract_key_points_8x_naive(float const*        prob,
                                                     int                 width,
                                                     int                 height,
                                                     int                 channel,
                                                     float               threshold,
                                                     int                 bkg_at_last,
                                                     unsigned int*       point_list)
{
    int i = 0, j = 0, k = 0;
    int valid_point_cnt  = 0;
    float const*      cur_prob_ptr  = prob;
    int               channel_class = channel;
    if(0 != bkg_at_last)
        channel_class = channel_class - 1;
    for(i = 0 ; i < height ; i ++)
    {
        for(j = 0 ; j < width ; j ++)
        {
            for(k = 0 ; k < channel_class ; k ++)
            {
                if(cur_prob_ptr[k] >= threshold)
                {
                    unsigned int                    x =  (j << 3) + (k & 7) ;
                    unsigned int                    y =  (i << 3) + (k >> 3) ;
                    point_list[2 * valid_point_cnt]               = (y << 16) | x;
                    ((float*)point_list)[2 * valid_point_cnt + 1] = cur_prob_ptr[k];
                    valid_point_cnt ++;
                }
            }
            cur_prob_ptr += channel;
        }
    }

    return valid_point_cnt;
}

static int   super_point_extract_key_points_naive(float const*        prob,
                                                  int                 width,
                                                  int                 height,
                                                  int                 channel,
                                                  int                 upsample_scale,
                                                  float               threshold,
                                                  int                 bkg_at_last,
                                                  unsigned int*       point_list)
{
    int i = 0, j = 0, k = 0;
    int valid_point_cnt  = 0;
    float const*      cur_prob_ptr  = prob;
    int               channel_class = channel;
    if(0 != bkg_at_last)
        channel_class = channel_class - 1;
    for(i = 0 ; i < height ; i ++)
    {
        for(j = 0 ; j < width ; j ++)
        {
            for(k = 0 ; k < channel_class ; k ++)
            {
                if(cur_prob_ptr[k] >= threshold)
                {
                    unsigned int x =  j * upsample_scale + (k % upsample_scale) ;
                    unsigned int y =  i * upsample_scale + (k / upsample_scale) ;
                    point_list[2 * valid_point_cnt]               = (y << 16) | x;
                    ((float*)point_list)[2 * valid_point_cnt + 1] = cur_prob_ptr[k];
                    valid_point_cnt ++;
                }
            }
            cur_prob_ptr += channel;
        }
    }

    return valid_point_cnt;
}

typedef struct tag_compact_point{
    unsigned short int   x;
    unsigned short int   y;
    float                score;
}COMPACT_POINT, *PCOMPACT_POINT;

static int kCmpValue[2] = { -1, 1 };
static int cmp_score(const void* left, const void* right)
{
    float left_val  = ((float*)left)[1];
    float right_val = ((float*)right)[1];
    return kCmpValue[left_val < right_val];
}
/*
static int cmp_score_full(const void* left, const void* right)
{
    DLCV_SP_KEY_POINT* left_pt  = ((DLCV_SP_KEY_POINT*)left);
    DLCV_SP_KEY_POINT* right_pt = ((DLCV_SP_KEY_POINT*)right);
    return kCmpValue[left_pt->reserved > right_pt->reserved];
}
*/
static void remove_mns_border(unsigned char*   nms_mask_map,
                              int              raw_width, 
                              int              raw_height, 
                              int              pad,
                              int              border_size)
{
    int  i          = 0, j = 0;
    int  map_w      = (raw_width  + 2 * pad);
    int  map_h      = (raw_height + 2 * pad);

    for(i = 0 ; i < border_size ; i ++)
    {
        memset(nms_mask_map + (pad + i) * map_w, 0, map_w);
        memset(nms_mask_map + (pad + raw_height - border_size + i) * map_w, 0, map_w);
    }
    for(i = pad + border_size ; i < (map_h - pad - border_size); i ++)
    {
        for(j = 0 ; j < border_size ; j ++)
        {
            nms_mask_map[i * map_w + pad + j]                           = 0;
            nms_mask_map[i * map_w + pad + raw_width - border_size + j] = 0;
        }
    }
}

//#define DEBUG_NMS_XY

static int select_nms_map(COMPACT_POINT const*  pt_list, 
                          int                   pt_cnt, 
                          unsigned char*        nms_mask_map, 
                          int                   raw_width, 
                          int                   raw_height, 
                          int                   pad,
                          DLCV_SP_KEY_POINT*    output_kpts,
                          int                   output_kpts_count)
{
    int   map_w        = 2 * pad + raw_width;   
    int   valid_pt_cnt = 0;
    int   i            = 0;
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        int   x            = pt_list[i].x + pad;
        int   y            = pt_list[i].y + pad;
        int   cur_mask_pos = y * map_w + x;
        int   cmp_res      = (255 == nms_mask_map[cur_mask_pos]) & (valid_pt_cnt < output_kpts_count);
        if(1 == cmp_res)
        {
            output_kpts[valid_pt_cnt].x        = pt_list[i].x;
            output_kpts[valid_pt_cnt].y        = pt_list[i].y;
            output_kpts[valid_pt_cnt].score    = pt_list[i].score;
            output_kpts[valid_pt_cnt].reserved = (unsigned int)(output_kpts[valid_pt_cnt].x + output_kpts[valid_pt_cnt].y * raw_width) ;
            valid_pt_cnt ++;
        }
    }

    return valid_pt_cnt;
}

static void super_point_nms_map(COMPACT_POINT const*  pt_list, 
                                int                   pt_cnt, 
                                unsigned char*        nms_mask_map, 
                                int                   map_w, 
                                int                   pad)
{
    int  i = 0, j = 0, k = 0;
    int  clean_size = 2 * pad + 1;
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        unsigned int  x    = pt_list[i].x ;
        unsigned int  y    = pt_list[i].y ;
        unsigned int  pos  = (y + pad) * map_w + x + pad;
        unsigned char flag = nms_mask_map[pos];
        if(1 == flag)
        {
            unsigned int  clean_y = y ;
            for(j = 0 ; j < clean_size ; j ++ )
            {
                unsigned int  clean_x = x ;
                for(k = 0 ; k < clean_size ; k ++)
                {
                    unsigned int  clean_pos = clean_y * map_w + clean_x;
                    nms_mask_map[clean_pos] = 0;
                    clean_x ++;
                }
                clean_y ++;
            }
            nms_mask_map[pos] = 255;
        }
    }
}

static void nms_map(COMPACT_POINT const*  pt_list, 
                    int                   pt_cnt, 
                    unsigned char*        nms_mask_map, 
                    int                   raw_width, 
                    int                   raw_height, 
                    int                   pad,
                    int                   border_size)
{
    int  i          = 0, j = 0, k = 0;
    int  map_w      = (raw_width  + 2 * pad);
    int  map_h      = (raw_height + 2 * pad);
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        unsigned int x = pt_list[i].x ;
        unsigned int y = pt_list[i].y ;
        nms_mask_map[(y + pad) * map_w + x + pad] = 1;
    }

    if(8 != pad)
        super_point_nms_map(pt_list, pt_cnt, nms_mask_map, map_w, pad);
    else
        super_point_nms_map_8x_implement((unsigned short int*)pt_list, pt_cnt, nms_mask_map, map_w);

    remove_mns_border(nms_mask_map, raw_width, raw_height, pad, border_size);
}

int   dlcv_super_point_extract_key_points(const float*        prob,
                                          int                 width,
                                          int                 height,
                                          int                 channel,
                                          int                 upsample_scale,
                                          float               threshold,
                                          int                 bkg_at_last,
                                          unsigned char*      assist_buf,
                                          DLCV_SP_KEY_POINT*  output_kpts,
                                          int                 output_kpts_count)
{
    int            i = 0, j = 0;
    int            channel_aligned  = (((channel + 3) >> 2) << 2);
    int            point_cnt        = width * height;
    int            total_valid_cnt  = 0;
    unsigned int   pt_flag[160]     = { 0 };
    unsigned int*  pt_list          = (unsigned int*)(assist_buf);
    unsigned int   nms_mask_map_len = (width * upsample_scale + 2 * upsample_scale) * (height * upsample_scale + 2 * upsample_scale);
    unsigned char* nms_mask         = (unsigned char*)(pt_list + (2 * width * upsample_scale * height * upsample_scale));
    int            border_size      = (upsample_scale >> 1) ;
    if(((upsample_scale * upsample_scale) > channel) ||
       (channel_aligned >= 160) ||
       (width <= 2) ||
       (height <= 2))
        return 0;
    
    if(8 == upsample_scale)
    {
        total_valid_cnt = super_point_extract_key_points_8x(prob, 
                                                            width, 
                                                            height, 
                                                            channel, 
                                                            threshold, 
                                                            bkg_at_last, 
                                                            pt_flag,
                                                            pt_list);
    }
    else
        total_valid_cnt = super_point_extract_key_points(prob, 
                                                         width, 
                                                         height, 
                                                         channel, 
                                                         upsample_scale, 
                                                         threshold, 
                                                         bkg_at_last, 
                                                         pt_flag,
                                                         pt_list);

    if(1 == total_valid_cnt && output_kpts_count >= total_valid_cnt)
    {
        unsigned int  xy     = *((unsigned int*)pt_list);
        output_kpts[0].x     = (int)(xy & 0x0000FFFFU);
        output_kpts[0].y     = (int)((xy & 0xFFFF0000U) >> 16);
        output_kpts[0].score = *((float*)(pt_list + 1));
        return 1;
    }
    if(total_valid_cnt <= 0)
        return 0;
    qsort(pt_list, total_valid_cnt, sizeof(COMPACT_POINT), cmp_score);
    memset(nms_mask, 0, nms_mask_map_len);
    nms_map((COMPACT_POINT*)pt_list,
            total_valid_cnt,
            nms_mask,
            width * upsample_scale,
            height * upsample_scale,
            upsample_scale,
            border_size);
    total_valid_cnt = select_nms_map((COMPACT_POINT*)pt_list, 
                                     total_valid_cnt, 
                                     nms_mask, 
                                     width * upsample_scale, 
                                     height * upsample_scale, 
                                     upsample_scale,
                                     output_kpts,
                                     output_kpts_count);

    //qsort(output_kpts, total_valid_cnt, sizeof(DLCV_SP_KEY_POINT), cmp_score_full);                                 
    return total_valid_cnt;
}

void   dlcv_super_point_normalize_descriptor(const float*    descriptor_map,
                                             int             descriptor_width,
                                             int             descriptor_height,
                                             int             descriptor_channel,
                                             float*          normalize_descriptor_map)
{
    normalize_super_point_feature_implement(descriptor_map, 
                                            descriptor_width, 
                                            descriptor_height, 
                                            descriptor_channel, 
                                            normalize_descriptor_map);
}

static void extract_super_point_feature(const DLCV_SP_KEY_POINT*    key_points,
                                        int                         key_points_count,
                                        const float*                normalize_feature_map,
                                        int                         feature_width,
                                        int                         feature_height,
                                        int                         feature_channel,
                                        int                         upsample_scale,
                                        float*                      result)
{
    int i = 0, j = 0;
    int ratio = 1.0f / (float)(upsample_scale);
    float u = 0.0f, v = 0.0f;
    for(i = 0 ; i < key_points_count ; i ++)
    {
        int           src_x      = key_points[i].x;
        int           src_y      = key_points[i].y;
        int           blk_x0     = src_x / upsample_scale;
        int           blk_y0     = src_y / upsample_scale;
        int           blk_x1     = blk_x0 + 1;
        int           blk_y1     = blk_y0 + 1;
        int           x_in_blk   = src_x % upsample_scale;
        int           y_in_blk   = src_y % upsample_scale;
        float const*  d00        = 0;
        float const*  d01        = 0;
        float const*  d10        = 0;
        float const*  d11        = 0;
        float         sum        = 0.0f;
        u                        = x_in_blk * ratio;
        v                        = y_in_blk * ratio;
        blk_x1                   = (blk_x1 >= feature_width)  ? (feature_width - 1)  : blk_x1;
        blk_y1                   = (blk_y1 >= feature_height) ? (feature_height - 1) : blk_y1;
        d00                      = normalize_feature_map + blk_y0 * feature_width * feature_channel + blk_x0 * feature_channel;
        d01                      = normalize_feature_map + blk_y0 * feature_width * feature_channel + blk_x1 * feature_channel;
        d10                      = normalize_feature_map + blk_y1 * feature_width * feature_channel + blk_x0 * feature_channel;
        d11                      = normalize_feature_map + blk_y1 * feature_width * feature_channel + blk_x1 * feature_channel;
        sum                      = extract_super_point_feature_implement(d00, d01, d10, d11, u, v, feature_channel, result);
        sum                      = sqrtf(sum);
        sum                      = sum < 1e-6 ? 1e-6 : sum;
        sum                      = 1.0f / sum;
        super_point_normalize_feature_in_place_implement(result, feature_channel, sum);
        result                  += feature_channel;
    }
}

void   dlcv_super_point_extract_point_descriptor(const DLCV_SP_KEY_POINT*    key_points,
                                                 int                         key_points_count,
                                                 const float*                descriptor_map,
                                                 int                         descriptor_width,
                                                 int                         descriptor_height,
                                                 int                         descriptor_channel,
                                                 int                         upsample_scale,
                                                 float*                      result)
{
    extract_super_point_feature(key_points, 
                                key_points_count, 
                                descriptor_map, 
                                descriptor_width, 
                                descriptor_height, 
                                descriptor_channel, 
                                upsample_scale, 
                                result);
}
