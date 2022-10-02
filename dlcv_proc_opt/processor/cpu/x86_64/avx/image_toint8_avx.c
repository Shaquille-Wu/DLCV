#include <immintrin.h>

void image_to_int8_f1uc1_aligned16_implement_avx(float const*      src,
                                                 unsigned char*    dst,
                                                 int               src_width,
                                                 int               src_height,
                                                 int               src_line_size,
                                                 int               dst_line_size,
                                                 float             scale,
                                                 float             bias)
{
    int           i           = 0;
    int           j           = 0;
    __m256        scale_value = _mm256_set1_ps(scale);
    __m256        bias_value  = _mm256_set1_ps(bias);
    __m256        min_value   = _mm256_set1_ps(0.0f);
    __m256        plus        = _mm256_set1_ps(0.5f);
    for (i = 0; i < src_height; i++)
    {
        float const*    src_ptr = src + i * src_line_size;
        unsigned char*  dst_ptr = dst + i * dst_line_size;
        for (j = 0; j < src_width; j += 16)
        {
            __m256    data0  = _mm256_loadu_ps(src_ptr);
            __m256    data1  = _mm256_loadu_ps(src_ptr + 8);

            data0 = _mm256_mul_ps(data0, scale_value);
            data1 = _mm256_mul_ps(data1, scale_value);

            data0 = _mm256_add_ps(data0, bias_value);
            data1 = _mm256_add_ps(data1, bias_value);

            data0 = _mm256_max_ps(data0, min_value);
            data1 = _mm256_max_ps(data1, min_value);

            data0 = _mm256_add_ps(data0, plus);
            data1 = _mm256_add_ps(data1, plus);
            __m256i cur_int8_data0 = _mm256_cvtps_epi32(_mm256_round_ps(data0, 3));
            __m256i cur_int8_data1 = _mm256_cvtps_epi32(_mm256_round_ps(data1, 3));

            __m256i cur_int8_data2 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 32);
            __m256i cur_int8_data3 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 49);
            cur_int8_data2    = _mm256_packus_epi32(cur_int8_data2, cur_int8_data3);
            cur_int8_data0    = _mm256_permute2x128_si256(cur_int8_data2, cur_int8_data2, 1);
            cur_int8_data2    = _mm256_packus_epi16(cur_int8_data2, cur_int8_data0);
            __m128i  res      = _mm256_extracti128_si256(cur_int8_data2, 0);

            _mm_storeu_si128((__m128i*)(dst_ptr), res);

            src_ptr += 16;
            dst_ptr += 16;
        }
    }
}

void image_to_int8_f1uc1_aligned32_implement_avx(float const*      src,
                                                 unsigned char*    dst,
                                                 int               src_width,
                                                 int               src_height,
                                                 int               src_line_size,
                                                 int               dst_line_size,
                                                 float             scale,
                                                 float             bias)
{
    int           i           = 0;
    int           j           = 0;
    __m256        scale_value = _mm256_set1_ps(scale);
    __m256        bias_value  = _mm256_set1_ps(bias);
    __m256        min_value   = _mm256_set1_ps(0.0f);
    __m256        plus        = _mm256_set1_ps(0.5f);
    for (i = 0; i < src_height; i++)
    {
        float const*    src_ptr = src + i * src_line_size;
        unsigned char*  dst_ptr = dst + i * dst_line_size;
        for (j = 0; j < src_width; j += 32)
        {
            __m256    data0  = _mm256_loadu_ps(src_ptr);
            __m256    data1  = _mm256_loadu_ps(src_ptr + 8);
            __m256    data2  = _mm256_loadu_ps(src_ptr + 16);
            __m256    data3  = _mm256_loadu_ps(src_ptr + 24);

            data0 = _mm256_mul_ps(data0, scale_value);
            data1 = _mm256_mul_ps(data1, scale_value);
            data2 = _mm256_mul_ps(data2, scale_value);
            data3 = _mm256_mul_ps(data3, scale_value);

            data0 = _mm256_add_ps(data0, bias_value);
            data1 = _mm256_add_ps(data1, bias_value);
            data2 = _mm256_add_ps(data2, bias_value);
            data3 = _mm256_add_ps(data3, bias_value);

            data0 = _mm256_max_ps(data0, min_value);
            data1 = _mm256_max_ps(data1, min_value);
            data2 = _mm256_max_ps(data2, min_value);
            data3 = _mm256_max_ps(data3, min_value);

            data0 = _mm256_add_ps(data0, plus);
            data1 = _mm256_add_ps(data1, plus);
            data2 = _mm256_add_ps(data2, plus);
            data3 = _mm256_add_ps(data3, plus);
            __m256i cur_int8_data0 = _mm256_cvtps_epi32(_mm256_round_ps(data0, 3));
            __m256i cur_int8_data1 = _mm256_cvtps_epi32(_mm256_round_ps(data1, 3));
            __m256i cur_int8_data2 = _mm256_cvtps_epi32(_mm256_round_ps(data2, 3));
            __m256i cur_int8_data3 = _mm256_cvtps_epi32(_mm256_round_ps(data3, 3));
            __m256i cur_int8_data4 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 32);
            __m256i cur_int8_data5 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 49);
            __m256i cur_int8_data6 = _mm256_permute2x128_si256(cur_int8_data2, cur_int8_data3, 32);
            __m256i cur_int8_data7 = _mm256_permute2x128_si256(cur_int8_data2, cur_int8_data3, 49);
            cur_int8_data0         = _mm256_packus_epi32(cur_int8_data4, cur_int8_data5);
            cur_int8_data1         = _mm256_packus_epi32(cur_int8_data6, cur_int8_data7);
            cur_int8_data2         = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 32);
            cur_int8_data3         = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 49);
            cur_int8_data4         = _mm256_packus_epi16(cur_int8_data2, cur_int8_data3);

            _mm256_storeu_si256((__m256i*)(dst_ptr), cur_int8_data4);

            src_ptr += 32;
            dst_ptr += 32;
        }
    }
}

void data_to_int8_aligned32_implement_avx(float const*      src,
                                          unsigned char*    dst,
                                          int               data_count,
                                          float             scale,
                                          float             bias)
{
    int             i           = 0;
    __m256          scale_value = _mm256_set1_ps(scale);
    __m256          bias_value  = _mm256_set1_ps(bias);
    __m256          min_value   = _mm256_set1_ps(0.0f);
    __m256          plus        = _mm256_set1_ps(0.5f);
    float const*    src_ptr     = src;
    unsigned char*  dst_ptr     = dst;

    for (i = 0; i < data_count; i += 32)
    {
        __m256    data0  = _mm256_loadu_ps(src_ptr);
        __m256    data1  = _mm256_loadu_ps(src_ptr + 8);
        __m256    data2  = _mm256_loadu_ps(src_ptr + 16);
        __m256    data3  = _mm256_loadu_ps(src_ptr + 24);

        data0 = _mm256_mul_ps(data0, scale_value);
        data1 = _mm256_mul_ps(data1, scale_value);
        data2 = _mm256_mul_ps(data2, scale_value);
        data3 = _mm256_mul_ps(data3, scale_value);

        data0 = _mm256_add_ps(data0, bias_value);
        data1 = _mm256_add_ps(data1, bias_value);
        data2 = _mm256_add_ps(data2, bias_value);
        data3 = _mm256_add_ps(data3, bias_value);

        data0 = _mm256_max_ps(data0, min_value);
        data1 = _mm256_max_ps(data1, min_value);
        data2 = _mm256_max_ps(data2, min_value);
        data3 = _mm256_max_ps(data3, min_value);

        data0 = _mm256_add_ps(data0, plus);
        data1 = _mm256_add_ps(data1, plus);
        data2 = _mm256_add_ps(data2, plus);
        data3 = _mm256_add_ps(data3, plus);
        __m256i cur_int8_data0 = _mm256_cvtps_epi32(_mm256_round_ps(data0, 3));
        __m256i cur_int8_data1 = _mm256_cvtps_epi32(_mm256_round_ps(data1, 3));
        __m256i cur_int8_data2 = _mm256_cvtps_epi32(_mm256_round_ps(data2, 3));
        __m256i cur_int8_data3 = _mm256_cvtps_epi32(_mm256_round_ps(data3, 3));
        __m256i cur_int8_data4 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 32);
        __m256i cur_int8_data5 = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 49);
        __m256i cur_int8_data6 = _mm256_permute2x128_si256(cur_int8_data2, cur_int8_data3, 32);
        __m256i cur_int8_data7 = _mm256_permute2x128_si256(cur_int8_data2, cur_int8_data3, 49);
        cur_int8_data0         = _mm256_packus_epi32(cur_int8_data4, cur_int8_data5);
        cur_int8_data1         = _mm256_packus_epi32(cur_int8_data6, cur_int8_data7);
        cur_int8_data2         = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 32);
        cur_int8_data3         = _mm256_permute2x128_si256(cur_int8_data0, cur_int8_data1, 49);
        cur_int8_data4         = _mm256_packus_epi16(cur_int8_data2, cur_int8_data3);

        _mm256_storeu_si256((__m256i*)(dst_ptr), cur_int8_data4);

        src_ptr += 32;
        dst_ptr += 32;
    }
}