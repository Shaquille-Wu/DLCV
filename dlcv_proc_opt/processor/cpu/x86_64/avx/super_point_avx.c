#include <immintrin.h>
#include "../../../../common/dlcv_proc_opt_com_def.h"
#include <math.h>

int super_point_extract_flag_implement_avx(float const*     prob, 
                                           int              channel, 
                                           int*             pt_flag,
                                           float            threshold)
{
    int            j               = 0;
    __m256         cur_thd         = _mm256_set1_ps(threshold);
    const __m256i  one             = _mm256_set1_epi32(1);
    int            channel_aligned = ((channel >> 3) << 3);
    int            valid_cnt       = 0;
    for(j = 0 ; j < channel_aligned ; j += 8)
    {
        __m256   cur_data    = _mm256_loadu_ps(prob);
        __m256   cur_mask    = _mm256_cmp_ps(cur_data, cur_thd, 13);
        __m256i  cur_mask_i  = _mm256_castps_si256(cur_mask);
        __m256i  cur_flag    = _mm256_and_si256(cur_mask_i, one);
        pt_flag[valid_cnt]   = j;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 0);
        pt_flag[valid_cnt]   = j + 1;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 1);
        pt_flag[valid_cnt]   = j + 2;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 2);
        pt_flag[valid_cnt]   = j + 3;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 3);
        pt_flag[valid_cnt]   = j + 4;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 4);
        pt_flag[valid_cnt]   = j + 5;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 5);
        pt_flag[valid_cnt]   = j + 6;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 6);
        pt_flag[valid_cnt]   = j + 7;
        valid_cnt           += _mm256_extract_epi32(cur_flag, 7);
        prob                += 8;
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

void normalize_super_point_feature_implement_avx(float const*    feature_map, 
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
        __m256   channel_sum_8    = _mm256_set1_ps(0.0f);
        float    channel_sum      = 0.0f;
        __m128   low, high;
        src_feature_ptr           = feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            __m256   cur_data     = _mm256_loadu_ps(src_feature_ptr);
            __m256   cur_data_sqr = _mm256_mul_ps(cur_data, cur_data);
            channel_sum_8         = _mm256_add_ps(channel_sum_8, cur_data_sqr);
            src_feature_ptr      += 8;
        }

        low          = _mm256_extractf128_ps(channel_sum_8, 0);
        high         = _mm256_extractf128_ps(channel_sum_8, 1);
        channel_sum += _mm_cvtss_f32(low);
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low),  4)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low),  8)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low), 12)));
        channel_sum += _mm_cvtss_f32(high);
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high),  4)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high),  8)));
        channel_sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high), 12)));

        for(; j < feature_channel ; j ++)
        {
            channel_sum     += (src_feature_ptr[0] * src_feature_ptr[0]);
            src_feature_ptr += 1;
        }
        
        low             = _mm_set1_ps(channel_sum);
        channel_sum     = _mm_cvtss_f32(_mm_max_ps(low, eps));
        channel_sum     = sqrtf(channel_sum);
        channel_sum     = 1.0f / channel_sum ;
        //channel_sum     = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set1_ps(channel_sum)));
        channel_sum_8   = _mm256_set1_ps(channel_sum);
        src_feature_ptr = feature_map;
        dst_feature_ptr = normalize_feature_map;
        for(j = 0 ; j < channel_aligned ; j += 8)
        {
            __m256   cur_data     = _mm256_loadu_ps(src_feature_ptr);
            __m256   cur_data_inv = _mm256_mul_ps(cur_data, channel_sum_8);
            _mm256_storeu_ps(dst_feature_ptr, cur_data_inv);
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


float extract_super_point_feature_implement_avx(float const*      feature_00, 
                                                float const*      feature_01, 
                                                float const*      feature_10, 
                                                float const*      feature_11, 
                                                float             u,
                                                float             v,
                                                int               feature_channel,
                                                float*            result)
{
    int           channel_aligned  = ((feature_channel >> 4) << 4);
    int           i                = 0 ;
    float         u1v1             = (1.0f - u) * (1.0f - v);
    float         u0v1             = u * (1.0f - v);
    float         u1v0             = (1.0f - u) * v;
    float         u0v0             = u * v;
    float         sum              = 0.0f;
    __m256        u1v1_coef        = _mm256_set1_ps(u1v1);
    __m256        u0v1_coef        = _mm256_set1_ps(u0v1);
    __m256        u1v0_coef        = _mm256_set1_ps(u1v0);
    __m256        u0v0_coef        = _mm256_set1_ps(u0v0);
    __m256        sum_8            = _mm256_set1_ps(0.0f);
    __m128        low, high;
    for(i = 0 ; i < channel_aligned ; i += 16)
    {
        __m256  d000  = _mm256_loadu_ps(feature_00);
        __m256  d001  = _mm256_loadu_ps(feature_01);
        __m256  d010  = _mm256_loadu_ps(feature_10);
        __m256  d011  = _mm256_loadu_ps(feature_11);
        __m256  d100  = _mm256_loadu_ps(feature_00 + 8);
        __m256  d101  = _mm256_loadu_ps(feature_01 + 8);
        __m256  d110  = _mm256_loadu_ps(feature_10 + 8);
        __m256  d111  = _mm256_loadu_ps(feature_11 + 8);

        __m256  res0  = _mm256_mul_ps(d000, u1v1_coef);
        __m256  res1  = _mm256_mul_ps(d001, u0v1_coef);
        __m256  res2  = _mm256_mul_ps(d010, u1v0_coef);
        __m256  res3  = _mm256_mul_ps(d011, u0v0_coef);
        res0          = _mm256_add_ps(res0, res1);
        res2          = _mm256_add_ps(res2, res3);
        res0          = _mm256_add_ps(res0, res2);

        _mm256_storeu_ps(result,     res0);
        res0          = _mm256_mul_ps(res0, res0);
        sum_8         = _mm256_add_ps(sum_8, res0);

        res0          = _mm256_mul_ps(d100, u1v1_coef);
        res1          = _mm256_mul_ps(d101, u0v1_coef);
        res2          = _mm256_mul_ps(d110, u1v0_coef);
        res3          = _mm256_mul_ps(d111, u0v0_coef);
        res0          = _mm256_add_ps(res0, res1);
        res2          = _mm256_add_ps(res2, res3);
        res0          = _mm256_add_ps(res0, res2);

        _mm256_storeu_ps(result + 8, res0);
        res0          = _mm256_mul_ps(res0, res0);
        sum_8         = _mm256_add_ps(sum_8, res0);

        feature_00  += 16;
        feature_01  += 16;
        feature_10  += 16;
        feature_11  += 16;
        result      += 16;
    }

    low          = _mm256_extractf128_ps(sum_8, 0);
    high         = _mm256_extractf128_ps(sum_8, 1);
    sum += _mm_cvtss_f32(low);
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low),  4)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low),  8)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(low), 12)));
    sum += _mm_cvtss_f32(high);
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high),  4)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high),  8)));
    sum += _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(high), 12)));

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

        res0         = res0 + res1;
        res2         = res2 + res3;
        res0         = res0 + res2;
        *result      = res0;
        sum         += res0 * res0;
        feature_00  += 1;
        feature_01  += 1;
        feature_10  += 1;
        feature_11  += 1;
        result      += 1;
    }

    return sum;
}

void normalize_super_point_feature_in_place_implement_avx(float* result, int feature_channel, float sum_rsqrt)
{
    int            j               = 0;
    int            channel_aligned = ((feature_channel >> 3) << 3);
    float const*   result_ptr      = result;
    __m256         sum_sqrt_inv_8  = _mm256_set1_ps(sum_rsqrt);
    for(j = 0 ; j < channel_aligned ; j += 8)
    {
        __m256   cur_data     = _mm256_loadu_ps(result);
        __m256   cur_data_inv = _mm256_mul_ps(cur_data, sum_sqrt_inv_8);
        _mm256_storeu_ps(result, cur_data_inv);
        result               += 8;
    }
    for(; j < feature_channel ; j ++)
    {
        float cur_data        = *result;
        float cur_data_inv    = cur_data * sum_rsqrt;
        *result               = cur_data_inv;
        result               += 1;
    }
}