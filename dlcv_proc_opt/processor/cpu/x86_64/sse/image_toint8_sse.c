#include <emmintrin.h>
#include <smmintrin.h>

void image_to_int8_f1uc1_aligned16_implement_sse(float const*      src,
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
    __m128        scale_value = _mm_set1_ps(scale);
    __m128        bias_value  = _mm_set1_ps(bias);
    __m128        min_value   = _mm_set1_ps(0.0f);
    __m128        plus        = _mm_set1_ps(0.5f);
    for (i = 0; i < src_height; i++)
    {
        float const*    src_ptr = src + i * src_line_size;
        unsigned char*  dst_ptr = dst + i * dst_line_size;
        for (j = 0; j < src_width; j += 16)
        {
            __m128   data0  = _mm_loadu_ps(src_ptr);
            __m128   data1  = _mm_loadu_ps(src_ptr + 4);
            __m128   data2  = _mm_loadu_ps(src_ptr + 8);
            __m128   data3  = _mm_loadu_ps(src_ptr + 12);

            data0 = _mm_mul_ps(data0, scale_value);
            data1 = _mm_mul_ps(data1, scale_value);
            data2 = _mm_mul_ps(data2, scale_value);
            data3 = _mm_mul_ps(data3, scale_value);

            data0 = _mm_add_ps(data0, bias_value);
            data1 = _mm_add_ps(data1, bias_value);
            data2 = _mm_add_ps(data2, bias_value);
            data3 = _mm_add_ps(data3, bias_value);

            data0 = _mm_max_ps(data0, min_value);
            data1 = _mm_max_ps(data1, min_value);
            data2 = _mm_max_ps(data2, min_value);
            data3 = _mm_max_ps(data3, min_value);

            data0     = _mm_add_ps(data0, plus);
            data1     = _mm_add_ps(data1, plus);
            data2     = _mm_add_ps(data2, plus);
            data3     = _mm_add_ps(data3, plus);
            __m128i cur_int8_data0 = _mm_cvtps_epi32(_mm_round_ps(data0, 3));
            __m128i cur_int8_data1 = _mm_cvtps_epi32(_mm_round_ps(data1, 3));
            __m128i cur_int8_data2 = _mm_cvtps_epi32(_mm_round_ps(data2, 3));
            __m128i cur_int8_data3 = _mm_cvtps_epi32(_mm_round_ps(data3, 3));

            cur_int8_data0 = _mm_packus_epi32(cur_int8_data0, cur_int8_data1);
            cur_int8_data2 = _mm_packus_epi32(cur_int8_data2, cur_int8_data3);
            cur_int8_data0 = _mm_packus_epi16(cur_int8_data0, cur_int8_data2);

            _mm_storeu_si128((__m128i*)dst_ptr, cur_int8_data0);

            src_ptr += 16;
            dst_ptr += 16;
        }
    }
}

void image_to_int8_f1uc1_aligned32_implement_sse(float const*      src,
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
    __m128        scale_value = _mm_set1_ps(scale);
    __m128        bias_value  = _mm_set1_ps(bias);
    __m128        min_value   = _mm_set1_ps(0.0f);
    __m128        plus        = _mm_set1_ps(0.5f);
    for (i = 0; i < src_height; i++)
    {
        float const*    src_ptr = src + i * src_line_size;
        unsigned char*  dst_ptr = dst + i * dst_line_size;
        for (j = 0; j < src_width; j += 32)
        {
            __m128   data0  = _mm_loadu_ps(src_ptr);
            __m128   data1  = _mm_loadu_ps(src_ptr + 4);
            __m128   data2  = _mm_loadu_ps(src_ptr + 8);
            __m128   data3  = _mm_loadu_ps(src_ptr + 12);
            __m128   data4  = _mm_loadu_ps(src_ptr + 16);
            __m128   data5  = _mm_loadu_ps(src_ptr + 20);
            __m128   data6  = _mm_loadu_ps(src_ptr + 24);
            __m128   data7  = _mm_loadu_ps(src_ptr + 28);

            data0 = _mm_mul_ps(data0, scale_value);
            data1 = _mm_mul_ps(data1, scale_value);
            data2 = _mm_mul_ps(data2, scale_value);
            data3 = _mm_mul_ps(data3, scale_value);
            data4 = _mm_mul_ps(data4, scale_value);
            data5 = _mm_mul_ps(data5, scale_value);
            data6 = _mm_mul_ps(data6, scale_value);
            data7 = _mm_mul_ps(data7, scale_value);

            data0 = _mm_add_ps(data0, bias_value);
            data1 = _mm_add_ps(data1, bias_value);
            data2 = _mm_add_ps(data2, bias_value);
            data3 = _mm_add_ps(data3, bias_value);
            data4 = _mm_add_ps(data4, bias_value);
            data5 = _mm_add_ps(data5, bias_value);
            data6 = _mm_add_ps(data6, bias_value);
            data7 = _mm_add_ps(data7, bias_value);

            data0 = _mm_max_ps(data0, min_value);
            data1 = _mm_max_ps(data1, min_value);
            data2 = _mm_max_ps(data2, min_value);
            data3 = _mm_max_ps(data3, min_value);
            data4 = _mm_max_ps(data4, min_value);
            data5 = _mm_max_ps(data5, min_value);
            data6 = _mm_max_ps(data6, min_value);
            data7 = _mm_max_ps(data7, min_value);

            data0     = _mm_add_ps(data0, plus);
            data1     = _mm_add_ps(data1, plus);
            data2     = _mm_add_ps(data2, plus);
            data3     = _mm_add_ps(data3, plus);
            data4     = _mm_add_ps(data4, plus);
            data5     = _mm_add_ps(data5, plus);
            data6     = _mm_add_ps(data6, plus);
            data7     = _mm_add_ps(data7, plus);
            __m128i cur_int8_data0 = _mm_cvtps_epi32(_mm_round_ps(data0, 3));
            __m128i cur_int8_data1 = _mm_cvtps_epi32(_mm_round_ps(data1, 3));
            __m128i cur_int8_data2 = _mm_cvtps_epi32(_mm_round_ps(data2, 3));
            __m128i cur_int8_data3 = _mm_cvtps_epi32(_mm_round_ps(data3, 3));
            __m128i cur_int8_data4 = _mm_cvtps_epi32(_mm_round_ps(data4, 3));
            __m128i cur_int8_data5 = _mm_cvtps_epi32(_mm_round_ps(data5, 3));
            __m128i cur_int8_data6 = _mm_cvtps_epi32(_mm_round_ps(data6, 3));
            __m128i cur_int8_data7 = _mm_cvtps_epi32(_mm_round_ps(data7, 3));

            cur_int8_data0 = _mm_packus_epi32(cur_int8_data0, cur_int8_data1);
            cur_int8_data2 = _mm_packus_epi32(cur_int8_data2, cur_int8_data3);
            cur_int8_data4 = _mm_packus_epi32(cur_int8_data4, cur_int8_data5);
            cur_int8_data6 = _mm_packus_epi32(cur_int8_data6, cur_int8_data7);

            cur_int8_data0 = _mm_packus_epi16(cur_int8_data0, cur_int8_data2);
            cur_int8_data4 = _mm_packus_epi16(cur_int8_data4, cur_int8_data6);

            _mm_storeu_si128((__m128i*)dst_ptr,        cur_int8_data0);
            _mm_storeu_si128((__m128i*)(dst_ptr + 16), cur_int8_data4);

            src_ptr += 32;
            dst_ptr += 32;
        }
    }
}

void data_to_int8_aligned32_implement_sse(float const*      src,
                                          unsigned char*    dst,
                                          int               data_count,
                                          float             scale,
                                          float             bias)
{
    int             i           = 0;
    __m128          scale_value = _mm_set1_ps(scale);
    __m128          bias_value  = _mm_set1_ps(bias);
    __m128          min_value   = _mm_set1_ps(0.0f);
    __m128          plus        = _mm_set1_ps(0.5f);
    float const*    src_ptr     = src;
    unsigned char*  dst_ptr     = dst;
    for (i = 0; i < data_count; i += 32)
    {
        __m128   data0  = _mm_loadu_ps(src_ptr);
        __m128   data1  = _mm_loadu_ps(src_ptr + 4);
        __m128   data2  = _mm_loadu_ps(src_ptr + 8);
        __m128   data3  = _mm_loadu_ps(src_ptr + 12);
        __m128   data4  = _mm_loadu_ps(src_ptr + 16);
        __m128   data5  = _mm_loadu_ps(src_ptr + 20);
        __m128   data6  = _mm_loadu_ps(src_ptr + 24);
        __m128   data7  = _mm_loadu_ps(src_ptr + 28);

        data0 = _mm_mul_ps(data0, scale_value);
        data1 = _mm_mul_ps(data1, scale_value);
        data2 = _mm_mul_ps(data2, scale_value);
        data3 = _mm_mul_ps(data3, scale_value);
        data4 = _mm_mul_ps(data4, scale_value);
        data5 = _mm_mul_ps(data5, scale_value);
        data6 = _mm_mul_ps(data6, scale_value);
        data7 = _mm_mul_ps(data7, scale_value);

        data0 = _mm_add_ps(data0, bias_value);
        data1 = _mm_add_ps(data1, bias_value);
        data2 = _mm_add_ps(data2, bias_value);
        data3 = _mm_add_ps(data3, bias_value);
        data4 = _mm_add_ps(data4, bias_value);
        data5 = _mm_add_ps(data5, bias_value);
        data6 = _mm_add_ps(data6, bias_value);
        data7 = _mm_add_ps(data7, bias_value);

        data0 = _mm_max_ps(data0, min_value);
        data1 = _mm_max_ps(data1, min_value);
        data2 = _mm_max_ps(data2, min_value);
        data3 = _mm_max_ps(data3, min_value);
        data4 = _mm_max_ps(data4, min_value);
        data5 = _mm_max_ps(data5, min_value);
        data6 = _mm_max_ps(data6, min_value);
        data7 = _mm_max_ps(data7, min_value);

        data0     = _mm_add_ps(data0, plus);
        data1     = _mm_add_ps(data1, plus);
        data2     = _mm_add_ps(data2, plus);
        data3     = _mm_add_ps(data3, plus);
        data4     = _mm_add_ps(data4, plus);
        data5     = _mm_add_ps(data5, plus);
        data6     = _mm_add_ps(data6, plus);
        data7     = _mm_add_ps(data7, plus);
        __m128i cur_int8_data0 = _mm_cvtps_epi32(_mm_round_ps(data0, 3));
        __m128i cur_int8_data1 = _mm_cvtps_epi32(_mm_round_ps(data1, 3));
        __m128i cur_int8_data2 = _mm_cvtps_epi32(_mm_round_ps(data2, 3));
        __m128i cur_int8_data3 = _mm_cvtps_epi32(_mm_round_ps(data3, 3));
        __m128i cur_int8_data4 = _mm_cvtps_epi32(_mm_round_ps(data4, 3));
        __m128i cur_int8_data5 = _mm_cvtps_epi32(_mm_round_ps(data5, 3));
        __m128i cur_int8_data6 = _mm_cvtps_epi32(_mm_round_ps(data6, 3));
        __m128i cur_int8_data7 = _mm_cvtps_epi32(_mm_round_ps(data7, 3));

        cur_int8_data0 = _mm_packus_epi32(cur_int8_data0, cur_int8_data1);
        cur_int8_data2 = _mm_packus_epi32(cur_int8_data2, cur_int8_data3);
        cur_int8_data4 = _mm_packus_epi32(cur_int8_data4, cur_int8_data5);
        cur_int8_data6 = _mm_packus_epi32(cur_int8_data6, cur_int8_data7);

        cur_int8_data0 = _mm_packus_epi16(cur_int8_data0, cur_int8_data2);
        cur_int8_data4 = _mm_packus_epi16(cur_int8_data4, cur_int8_data6);

        _mm_storeu_si128((__m128i*)dst_ptr,        cur_int8_data0);
        _mm_storeu_si128((__m128i*)(dst_ptr + 16), cur_int8_data4);

        src_ptr += 32;
        dst_ptr += 32;
    }
}