#include <emmintrin.h>
#include <smmintrin.h>
#include "../../../../common/dlcv_proc_opt_com_def.h"
#include <math.h>

int super_point_extract_flag_implement_sse(float const*     prob, 
                                           int              channel, 
                                           int*             pt_flag,
                                           float            threshold)
{
    int            j               = 0;
    __m128         cur_thd         = _mm_set1_ps(threshold);
    const __m128i  one             = _mm_set1_epi32(1);
    int            channel_aligned = ((channel >> 2) << 2);
    int            valid_cnt       = 0;
    for(j = 0 ; j < channel_aligned ; j += 4)
    {
        __m128   cur_data    = _mm_loadu_ps(prob);
        __m128   cur_mask    = _mm_cmpge_ps(cur_data, cur_thd);
        __m128i  cur_mask_i  = _mm_castps_si128(cur_mask);
        __m128i  cur_flag    = _mm_and_si128(cur_mask_i, one);
        pt_flag[valid_cnt]   = j;
        valid_cnt           += _mm_extract_epi32(cur_flag, 0);
        pt_flag[valid_cnt]   = j + 1;
        valid_cnt           += _mm_extract_epi32(cur_flag, 1);
        pt_flag[valid_cnt]   = j + 2;
        valid_cnt           += _mm_extract_epi32(cur_flag, 2);
        pt_flag[valid_cnt]   = j + 3;
        valid_cnt           += _mm_extract_epi32(cur_flag, 3);
        prob                += 4;
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

    return valid_cnt;
}

void normalize_super_point_feature_implement_sse(float const*    feature_map, 
                                                 int             feature_width, 
                                                 int             feature_height, 
                                                 int             feature_channel, 
                                                 float*          normalize_feature_map)
{
    int            i               = 0;
    int            j               = 0;
    int            img_size        = feature_width * feature_height;
    int            channel_aligned = ((feature_channel >> 3) << 3);
    float const*   src_feature_ptr = feature_map;
    float*         dst_feature_ptr = normalize_feature_map;
    __m128         eps             = _mm_set1_ps(1e-6f);
    for(i = 0 ; i < img_size ; i ++)
    {
        __m128   channel_sum_4    = _mm_set1_ps(0.0f);
        float    channel_sum      = 0.0f;
        src_feature_ptr           = feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            __m128   cur_data0     = _mm_loadu_ps(src_feature_ptr);
            __m128   cur_data1     = _mm_loadu_ps(src_feature_ptr + 4);
            __m128   cur_data_sqr0 = _mm_mul_ps(cur_data0, cur_data0);
            __m128   cur_data_sqr1 = _mm_mul_ps(cur_data1, cur_data1);
            channel_sum_4          = _mm_add_ps(channel_sum_4, cur_data_sqr0);
            channel_sum_4          = _mm_add_ps(channel_sum_4, cur_data_sqr1);
            src_feature_ptr       += 8;
        }
        channel_sum += _mm_cvtss_f32(channel_sum_4);
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(channel_sum_4),  4)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(channel_sum_4),  8)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(channel_sum_4), 12)));
        for(; j < feature_channel ; j ++)
        {
            channel_sum     += (src_feature_ptr[0] * src_feature_ptr[0]);
            src_feature_ptr += 1;
        }
        channel_sum_4   = _mm_set1_ps(channel_sum);
        channel_sum     = _mm_cvtss_f32(_mm_max_ps(channel_sum_4, eps));
        //channel_sum     = (channel_sum < 1e-6f ? 1e-6f : channel_sum);
        channel_sum     = sqrtf(channel_sum);
        channel_sum     = 1.0f / channel_sum ;
        //channel_sum     = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set1_ps(channel_sum)));
        channel_sum_4   = _mm_set1_ps(channel_sum);
        src_feature_ptr = feature_map;
        dst_feature_ptr = normalize_feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            __m128   cur_data0     = _mm_loadu_ps(src_feature_ptr);
            __m128   cur_data1     = _mm_loadu_ps(src_feature_ptr + 4);
            __m128   cur_data_inv0 = _mm_mul_ps(cur_data0, channel_sum_4);
            __m128   cur_data_inv1 = _mm_mul_ps(cur_data1, channel_sum_4);
            _mm_storeu_ps(dst_feature_ptr,     cur_data_inv0);
            _mm_storeu_ps(dst_feature_ptr + 4, cur_data_inv1);
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

float extract_super_point_feature_implement_sse(float const*      feature_00, 
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
    __m128        u1v1_coef        = _mm_set1_ps(u1v1);
    __m128        u0v1_coef        = _mm_set1_ps(u0v1);
    __m128        u1v0_coef        = _mm_set1_ps(u1v0);
    __m128        u0v0_coef        = _mm_set1_ps(u0v0);
    __m128        sum_4            = _mm_set1_ps(0.0f);
    for(i = 0 ; i < channel_aligned ; i += 8)
    {
        __m128  d000  = _mm_loadu_ps(feature_00);
        __m128  d001  = _mm_loadu_ps(feature_01);
        __m128  d010  = _mm_loadu_ps(feature_10);
        __m128  d011  = _mm_loadu_ps(feature_11);
        __m128  d100  = _mm_loadu_ps(feature_00 + 4);
        __m128  d101  = _mm_loadu_ps(feature_01 + 4);
        __m128  d110  = _mm_loadu_ps(feature_10 + 4);
        __m128  d111  = _mm_loadu_ps(feature_11 + 4);

        __m128  res0  = _mm_mul_ps(d000, u1v1_coef);
        __m128  res1  = _mm_mul_ps(d001, u0v1_coef);
        __m128  res2  = _mm_mul_ps(d010, u1v0_coef);
        __m128  res3  = _mm_mul_ps(d011, u0v0_coef);
        res0          = _mm_add_ps(res0, res1);
        res2          = _mm_add_ps(res2, res3);
        res0          = _mm_add_ps(res0, res2);

        _mm_storeu_ps(result,     res0);

        res0          = _mm_mul_ps(res0, res0);
        sum_4         = _mm_add_ps(sum_4, res0);

        res0          = _mm_mul_ps(d100, u1v1_coef);
        res1          = _mm_mul_ps(d101, u0v1_coef);
        res2          = _mm_mul_ps(d110, u1v0_coef);
        res3          = _mm_mul_ps(d111, u0v0_coef);
        res0          = _mm_add_ps(res0, res1);
        res2          = _mm_add_ps(res2, res3);
        res0          = _mm_add_ps(res0, res2);

        _mm_storeu_ps(result + 4, res0);

        res0          = _mm_mul_ps(res0, res0);
        sum_4         = _mm_add_ps(sum_4, res0);

        feature_00  += 8;
        feature_01  += 8;
        feature_10  += 8;
        feature_11  += 8;
        result      += 8;
    }

    sum += _mm_cvtss_f32(sum_4);
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_4),  4)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_4),  8)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_4), 12)));

    for(; i < feature_channel ; i ++)
    {
        float  d00  = *feature_00;
        float  d01  = *feature_01;
        float  d10  = *feature_10;
        float  d11  = *feature_11;

        float  res0 = d00 * u1v1;
        float  res1 = d01 * u0v1;
        float  res2 = d10 * u1v0;
        float  res3 = d11 * u0v0;

        res0        = res0 + res1;
        res2        = res2 + res3;
        res0        = res0 + res2;
        *result     = res0;

        sum         += res0 * res0;

        feature_00  += 1;
        feature_01  += 1;
        feature_10  += 1;
        feature_11  += 1;
        result      += 1;
    }

    return sum;
}

void normalize_super_point_feature_in_place_implement_sse(float* result, int feature_channel, float sum_rsqrt)
{
    int            j               = 0;
    int            channel_aligned = ((feature_channel >> 3) << 3);
    float const*   result_ptr      = result;
    __m128         sum_sqrt_inv_4  = _mm_set1_ps(sum_rsqrt);
    for(j = 0 ; j < channel_aligned ; j += 8)
    {
        __m128   cur_data0     = _mm_loadu_ps(result);
        __m128   cur_data1     = _mm_loadu_ps(result + 4);
        __m128   cur_data_inv0 = _mm_mul_ps(cur_data0, sum_sqrt_inv_4);
        __m128   cur_data_inv1 = _mm_mul_ps(cur_data1, sum_sqrt_inv_4);
        _mm_storeu_ps(result,     cur_data_inv0);
        _mm_storeu_ps(result + 4, cur_data_inv1);
        result      += 8;
    }
    for(; j < feature_channel ; j ++)
    {
        float cur_data        = *result;
        float cur_data_inv    = cur_data * sum_rsqrt;
        *result               = cur_data_inv;
    }
}