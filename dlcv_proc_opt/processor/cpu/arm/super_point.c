#include <arm_neon.h>

int super_point_extract_flag_implement(float const*     prob, 
                                       int              channel, 
                                       int*             pt_flag,
                                       float            threshold)
{
    int               j               = 0;
    float32x4_t       cur_thd         = vdupq_n_f32(threshold);  
    unsigned int      valid_cnt       = 0;
    const uint32x4_t  one             = vdupq_n_u32(1);
    int               channel_aligned = ((channel >> 2) << 2);
    for(j = 0 ; j < channel_aligned ; j += 4)
    {
        float32x4_t   cur_data    = vld1q_f32(prob);
        uint32x4_t    cur_mask    = vcgeq_f32(cur_data, cur_thd);
        uint32x4_t    cur_flag    = vandq_u32(cur_mask, one);
        pt_flag[valid_cnt]        = j;
        valid_cnt                += vgetq_lane_u32(cur_flag, 0);
        pt_flag[valid_cnt]        = j + 1;
        valid_cnt                += vgetq_lane_u32(cur_flag, 1);
        pt_flag[valid_cnt]        = j + 2;
        valid_cnt                += vgetq_lane_u32(cur_flag, 2);
        pt_flag[valid_cnt]        = j + 3;
        valid_cnt                += vgetq_lane_u32(cur_flag, 3);
        prob                     += 4;
    }

    for(; j < channel ; j ++)
    {
        if(prob[0] >= threshold)
        {
            pt_flag[valid_cnt] = j;
            valid_cnt ++;
        }
        prob += 1;
    }

    return (int)valid_cnt;
}

static void normalize_super_point_feature_aligned(float const*    feature_map, 
                                                  int             feature_size,  
                                                  int             feature_channel, 
                                                  float*          normalize_feature_map)
{
    int            i               = 0;
    int            j               = 0;
    float const*   src_feature_ptr = feature_map;
    float*         dst_feature_ptr = normalize_feature_map;
    float32x4_t    eps             = vdupq_n_f32(1e-6f);
    for(i = 0 ; i < feature_size ; i ++)
    {
        float32x4_t   channel_sum_4    = vdupq_n_f32(0.0f);
        float32x4_t   r;
        float         channel_sum      = 0.0f;
        src_feature_ptr           = feature_map;
        for(j = 0 ; j < feature_channel ; j += 8)
        {
            float32x4_t   cur_data0     = vld1q_f32(src_feature_ptr);
            float32x4_t   cur_data1     = vld1q_f32(src_feature_ptr + 4);
            float32x4_t   cur_data_sqr0 = vmulq_f32(cur_data0, cur_data0);
            float32x4_t   cur_data_sqr1 = vmulq_f32(cur_data1, cur_data1);
            channel_sum_4               = vaddq_f32(channel_sum_4, cur_data_sqr0);
            channel_sum_4               = vaddq_f32(channel_sum_4, cur_data_sqr1);
            src_feature_ptr            += 8;
        }
        channel_sum    += vgetq_lane_f32(channel_sum_4, 0);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 1);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 2);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 3);
        channel_sum_4   = vdupq_n_f32(channel_sum);
        channel_sum     = vgetq_lane_f32(vmaxq_f32(channel_sum_4, eps), 0);
        channel_sum_4   = vdupq_n_f32(channel_sum);
        r               = vrsqrteq_f32(channel_sum_4);
        r               = vmulq_f32(r, vrsqrtsq_f32(channel_sum_4, vmulq_f32(r, r)));
        r               = vmulq_f32(r, vrsqrtsq_f32(channel_sum_4, vmulq_f32(r, r)));
        channel_sum     = vgetq_lane_f32(r, 0);
        channel_sum_4   = vdupq_n_f32(channel_sum);
        src_feature_ptr = feature_map;
        dst_feature_ptr = normalize_feature_map;
        for(j = 0 ; j < feature_channel ; j += 8)
        {
            float32x4_t   cur_data0     = vld1q_f32(src_feature_ptr);
            float32x4_t   cur_data1     = vld1q_f32(src_feature_ptr + 4);
            float32x4_t   cur_data_inv0 = vmulq_f32(cur_data0, channel_sum_4);
            float32x4_t   cur_data_inv1 = vmulq_f32(cur_data1, channel_sum_4);
            vst1q_f32 (dst_feature_ptr,     cur_data_inv0);
            vst1q_f32 (dst_feature_ptr + 4, cur_data_inv1);
            src_feature_ptr           += 8;
            dst_feature_ptr           += 8;
        }
        feature_map           += feature_channel;
        normalize_feature_map += feature_channel;
    }
}

static void normalize_super_point_feature_unaligned(float const*    feature_map, 
                                                    int             feature_size,  
                                                    int             feature_channel, 
                                                    float*          normalize_feature_map)
{
    int            i               = 0;
    int            j               = 0;
    float const*   src_feature_ptr = feature_map;
    float*         dst_feature_ptr = normalize_feature_map;
    int            channel_aligned = ((feature_channel >> 3) << 3);
    float32x4_t    eps             = vdupq_n_f32(1e-6f);
    for(i = 0 ; i < feature_size ; i ++)
    {
        float32x4_t   channel_sum_4    = vdupq_n_f32(0.0f);
        float32x4_t   r;
        float         channel_sum      = 0.0f;
        src_feature_ptr                = feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            float32x4_t   cur_data0     = vld1q_f32(src_feature_ptr);
            float32x4_t   cur_data1     = vld1q_f32(src_feature_ptr + 4);
            float32x4_t   cur_data_sqr0 = vmulq_f32(cur_data0, cur_data0);
            float32x4_t   cur_data_sqr1 = vmulq_f32(cur_data1, cur_data1);
            channel_sum_4               = vaddq_f32(channel_sum_4, cur_data_sqr0);
            channel_sum_4               = vaddq_f32(channel_sum_4, cur_data_sqr1);
            src_feature_ptr            += 8;
        }
        channel_sum    += vgetq_lane_f32(channel_sum_4, 0);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 1);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 2);
        channel_sum    += vgetq_lane_f32(channel_sum_4, 3);
        for(; j < feature_channel ; j ++)
        {
            channel_sum     += (src_feature_ptr[0] * src_feature_ptr[0]);
            src_feature_ptr += 1;
        }
        channel_sum_4   = vdupq_n_f32(channel_sum);
        channel_sum     = vgetq_lane_f32(vmaxq_f32(channel_sum_4, eps), 0);
        channel_sum_4   = vdupq_n_f32(channel_sum);
        r               = vrsqrteq_f32(channel_sum_4);
        r               = vmulq_f32(r, vrsqrtsq_f32(channel_sum_4, vmulq_f32(r, r)));
        r               = vmulq_f32(r, vrsqrtsq_f32(channel_sum_4, vmulq_f32(r, r)));
        channel_sum     = vgetq_lane_f32(r, 0);
        channel_sum_4   = vdupq_n_f32(channel_sum);
        src_feature_ptr = feature_map;
        dst_feature_ptr = normalize_feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            float32x4_t   cur_data0     = vld1q_f32(src_feature_ptr);
            float32x4_t   cur_data1     = vld1q_f32(src_feature_ptr + 4);
            float32x4_t   cur_data_inv0 = vmulq_f32(cur_data0, channel_sum_4);
            float32x4_t   cur_data_inv1 = vmulq_f32(cur_data1, channel_sum_4);
            vst1q_f32 (dst_feature_ptr,     cur_data_inv0);
            vst1q_f32 (dst_feature_ptr + 4, cur_data_inv1);
            src_feature_ptr      += 8;
            dst_feature_ptr      += 8;
        }
        for(; j < feature_channel ; j ++)
        {
            float cur_data        = *src_feature_ptr;
            float cur_data_inv    = cur_data * channel_sum;
            *dst_feature_ptr      = cur_data_inv;
            src_feature_ptr      += 1;
            dst_feature_ptr      += 1;
        }
        feature_map           += feature_channel;
        normalize_feature_map += feature_channel;
    }
}

void normalize_super_point_feature_implement(float const*    feature_map, 
                                             int             feature_width, 
                                             int             feature_height, 
                                             int             feature_channel, 
                                             float*          normalize_feature_map)
{
    int    img_size = feature_width * feature_height;
    if(0 == (feature_channel & 7))
        normalize_super_point_feature_aligned(feature_map, 
                                              img_size, 
                                              feature_channel, 
                                              normalize_feature_map);
    else
        normalize_super_point_feature_unaligned(feature_map, 
                                                img_size, 
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
    int           channel_aligned  = ((feature_channel >> 3) << 3);
    int           i                = 0 ;
    float         u1v1             = (1.0f - u) * (1.0f - v);
    float         u0v1             = u * (1.0f - v);
    float         u1v0             = (1.0f - u) * v;
    float         u0v0             = u * v;
    float         sum              = 0.0f;
    float32x4_t   u1v1_coef        = vdupq_n_f32(u1v1);
    float32x4_t   u0v1_coef        = vdupq_n_f32(u0v1);
    float32x4_t   u1v0_coef        = vdupq_n_f32(u1v0);
    float32x4_t   u0v0_coef        = vdupq_n_f32(u0v0);
    float32x4_t   sum_4            = vdupq_n_f32(0.0f);
    for(i = 0 ; i < channel_aligned ; i += 8)
    {
        float32x4_t  d000  = vld1q_f32(feature_00);
        float32x4_t  d001  = vld1q_f32(feature_01);
        float32x4_t  d100  = vld1q_f32(feature_00 + 4);
        float32x4_t  d101  = vld1q_f32(feature_01 + 4);
        float32x4_t  res0  = vmulq_f32(d000, u1v1_coef);
        float32x4_t  res1  = vmulq_f32(d001, u0v1_coef);
        res0               = vaddq_f32(res0, res1);

        float32x4_t  d010  = vld1q_f32(feature_10);
        float32x4_t  d011  = vld1q_f32(feature_11);
        float32x4_t  d110  = vld1q_f32(feature_10 + 4);
        float32x4_t  d111  = vld1q_f32(feature_11 + 4);
        float32x4_t  res2  = vmulq_f32(d010, u1v0_coef);
        float32x4_t  res3  = vmulq_f32(d011, u0v0_coef);
        res2               = vaddq_f32(res2, res3);
        res0               = vaddq_f32(res0, res2);

        vst1q_f32(result,     res0);
        res0          = vmulq_f32(res0, res0);
        sum_4         = vaddq_f32(res0, sum_4);

        res0          = vmulq_f32(d100, u1v1_coef);
        res1          = vmulq_f32(d101, u0v1_coef);
        res2          = vmulq_f32(d110, u1v0_coef);
        res3          = vmulq_f32(d111, u0v0_coef);
        res0          = vaddq_f32(res0, res1);
        res2          = vaddq_f32(res2, res3);
        res0          = vaddq_f32(res0, res2);

        vst1q_f32(result + 4, res0);
        res0          = vmulq_f32(res0, res0);
        sum_4         = vaddq_f32(res0, sum_4);

        feature_00   += 8;
        feature_01   += 8;
        feature_10   += 8;
        feature_11   += 8;
        result       += 8;
    }

    sum += vgetq_lane_f32(sum_4, 0);
    sum += vgetq_lane_f32(sum_4, 1);
    sum += vgetq_lane_f32(sum_4, 2);
    sum += vgetq_lane_f32(sum_4, 3);

    for(; i < feature_channel ; i ++)
    {
        float  d00   = *feature_00;
        float  d01   = *feature_01;
        float  d10   = *feature_10;
        float  d11   = *feature_11;

        float  res0  = d00 * u1v1;
        float  res1  = d01 * u0v1;
        float  res2  = d10 * u1v0;
        float  res3  = d11 * u0v0;

        res0         = res0 + res1;
        res2         = res2 + res3;
        res0         = res0 + res2;
        *result      = res0;

        sum         += (res0 * res0);

        feature_00  += 1;
        feature_01  += 1;
        feature_10  += 1;
        feature_11  += 1;
        result      += 1;
    }

    return sum;
}

void super_point_normalize_feature_in_place_implement(float* result, int feature_channel, float sum_rsqrt)
{
    int            j               = 0;
    int            channel_aligned = ((feature_channel >> 3) << 3);
    float const*   result_ptr      = result;
    float32x4_t    sum_sqrt_inv_4  = vdupq_n_f32(sum_rsqrt);
    for(j = 0 ; j < channel_aligned ; j += 8)
    {
        float32x4_t   cur_data0     = vld1q_f32(result);
        float32x4_t   cur_data1     = vld1q_f32(result + 4);
        float32x4_t   cur_data_inv0 = vmulq_f32(cur_data0, sum_sqrt_inv_4);
        float32x4_t   cur_data_inv1 = vmulq_f32(cur_data1, sum_sqrt_inv_4);
        vst1q_f32(result,     cur_data_inv0);
        vst1q_f32(result + 4, cur_data_inv1);
        result      += 8;
    }
    for(; j < feature_channel ; j ++)
    {
        float cur_data        = *result;
        float cur_data_inv    = cur_data * sum_rsqrt;
        *result               = cur_data_inv;
    }
}

void super_point_nms_map_8x_implement(unsigned short int const*  pt_list, 
                                      int                        pt_cnt, 
                                      unsigned char*             nms_mask_map, 
                                      int                        map_w)
{
    int  i = 0, j = 0;
    int          clean_size = 17;
    uint32x4_t   zero       = vdupq_n_u32(0);
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
                vst1q_u32((unsigned int*)(nms_mask_map + clean_pos), zero);
                nms_mask_map[clean_pos + 16] = 0;
                clean_y ++;
            }
            nms_mask_map[pos] = 255;
        }
        pt_list += 4;
    }
}