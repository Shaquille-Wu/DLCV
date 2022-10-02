#include <emmintrin.h>
#include <smmintrin.h>

void normalize_image_uc1f1_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         float                 mean, 
                                         float                 inv_std)
{
    int           i           = 0;
    int           j           = 0;
    __m128        mean0       = _mm_set1_ps(mean);
    __m128        inv_std0    = _mm_set1_ps(inv_std);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr = src + i * src_line_element;
        float*                dst_ptr = dst + i * dst_line_element;
        for (j = 0; j < width_aligned; j += 16)
        {
            __m128i   int8_data_raw   = _mm_loadu_si128((__m128i*)src_ptr);
            __m128i   cur_int8_data00 = _mm_cvtepu8_epi32(int8_data_raw);
            __m128i   cur_int8_data01 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw, 4));
            __m128i   cur_int8_data10 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw, 8));
            __m128i   cur_int8_data11 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw, 12));
            __m128    data0           = _mm_cvtepi32_ps(cur_int8_data00);
            __m128    data1           = _mm_cvtepi32_ps(cur_int8_data01);
            __m128    data2           = _mm_cvtepi32_ps(cur_int8_data10);
            __m128    data3           = _mm_cvtepi32_ps(cur_int8_data11);

            data0 = _mm_sub_ps(data0, mean0);
            data1 = _mm_sub_ps(data1, mean0);
            data2 = _mm_sub_ps(data2, mean0);
            data3 = _mm_sub_ps(data3, mean0);

            data0 = _mm_mul_ps(data0, inv_std0);
            data1 = _mm_mul_ps(data1, inv_std0);
            data2 = _mm_mul_ps(data2, inv_std0);
            data3 = _mm_mul_ps(data3, inv_std0);

            _mm_storeu_ps(dst_ptr,      data0);
            _mm_storeu_ps(dst_ptr + 4,  data1);
            _mm_storeu_ps(dst_ptr + 8,  data2);
            _mm_storeu_ps(dst_ptr + 12, data3);

            src_ptr += 16;
            dst_ptr += 16;
        }
    }
}

void normalize_image_uc3f3_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         const float*          mean, 
                                         const float*          inv_std)
{
    int           i                = 0;
    int           j                = 0;
    int           linesize_aligned = width_aligned * 3;
    __m128        mean0            = _mm_loadu_ps(mean);
    __m128        mean1            = _mm_loadu_ps(mean + 4);
    __m128        mean2            = _mm_loadu_ps(mean + 8);
    __m128        inv_std0         = _mm_loadu_ps(inv_std);
    __m128        inv_std1         = _mm_loadu_ps(inv_std + 4);
    __m128        inv_std2         = _mm_loadu_ps(inv_std + 8);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr  = src + i * src_line_element;
        float*                dst_ptr  = dst + i * dst_line_element;
        for (j = 0; j < linesize_aligned; j += 24)
        {
            __m128i        int8_data_raw0  = _mm_loadu_si128((__m128i*)src_ptr);
            long long int  int64_data      = *((long long int*)(src_ptr + 16));
            __m64          int64_data_raw  = _m_from_int64 (int64_data);
            __m128i        int8_data_raw2  = _mm_movpi64_epi64(int64_data_raw);

            __m128i        cur_int8_data00 = _mm_cvtepu8_epi32(int8_data_raw0);
            __m128i        cur_int8_data01 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 4));
            __m128i        cur_int8_data10 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 8));
            __m128i        cur_int8_data11 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 12));
            __m128i        cur_int8_data20 = _mm_cvtepu8_epi32(int8_data_raw2);
            __m128i        cur_int8_data21 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 4));

            __m128         data0           = _mm_cvtepi32_ps(cur_int8_data00);
            __m128         data1           = _mm_cvtepi32_ps(cur_int8_data01);
            __m128         data2           = _mm_cvtepi32_ps(cur_int8_data10);
            __m128         data3           = _mm_cvtepi32_ps(cur_int8_data11);
            __m128         data4           = _mm_cvtepi32_ps(cur_int8_data20);
            __m128         data5           = _mm_cvtepi32_ps(cur_int8_data21);

            data0 = _mm_sub_ps(data0, mean0);
            data1 = _mm_sub_ps(data1, mean1);
            data2 = _mm_sub_ps(data2, mean2);
            data3 = _mm_sub_ps(data3, mean0);
            data4 = _mm_sub_ps(data4, mean1);
            data5 = _mm_sub_ps(data5, mean2);

            data0 = _mm_mul_ps(data0, inv_std0);
            data1 = _mm_mul_ps(data1, inv_std1);
            data2 = _mm_mul_ps(data2, inv_std2);
            data3 = _mm_mul_ps(data3, inv_std0);
            data4 = _mm_mul_ps(data4, inv_std1);
            data5 = _mm_mul_ps(data5, inv_std2);

            _mm_storeu_ps(dst_ptr,      data0);
            _mm_storeu_ps(dst_ptr + 4,  data1);
            _mm_storeu_ps(dst_ptr + 8,  data2);
            _mm_storeu_ps(dst_ptr + 12, data3);
            _mm_storeu_ps(dst_ptr + 16, data4);
            _mm_storeu_ps(dst_ptr + 20, data5);

            src_ptr += 24;
            dst_ptr += 24;
        }
    }
}

void normalize_image_uc4f4_implement_sse(const unsigned char*  src, 
                                         float*                dst, 
                                         int                   width_aligned, 
                                         int                   height, 
                                         int                   src_line_element, 
                                         int                   dst_line_element, 
                                         const float*          mean, 
                                         const float*          inv_std)
{
    int           i                = 0;
    int           j                = 0;
    int           linesize_aligned = width_aligned * 4;
    __m128        mean0            = _mm_loadu_ps(mean);
    __m128        inv_std0         = _mm_loadu_ps(inv_std);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr  = src + i * src_line_element;
        float*                dst_ptr  = dst + i * dst_line_element;
        for (j = 0; j < linesize_aligned; j += 32)
        {
            __m128i        int8_data_raw0  = _mm_loadu_si128((__m128i*)src_ptr);
			__m128i        int8_data_raw2  = _mm_loadu_si128((__m128i*)(src_ptr + 16));

            __m128i        cur_int8_data00 = _mm_cvtepu8_epi32(int8_data_raw0);
            __m128i        cur_int8_data01 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 4));
            __m128i        cur_int8_data10 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 8));
            __m128i        cur_int8_data11 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 12));
            __m128i        cur_int8_data20 = _mm_cvtepu8_epi32(int8_data_raw2);
            __m128i        cur_int8_data21 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 4));
            __m128i        cur_int8_data30 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 8));
            __m128i        cur_int8_data31 = _mm_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 12));

            __m128         data0           = _mm_cvtepi32_ps(cur_int8_data00);
            __m128         data1           = _mm_cvtepi32_ps(cur_int8_data01);
            __m128         data2           = _mm_cvtepi32_ps(cur_int8_data10);
            __m128         data3           = _mm_cvtepi32_ps(cur_int8_data11);
            __m128         data4           = _mm_cvtepi32_ps(cur_int8_data20);
            __m128         data5           = _mm_cvtepi32_ps(cur_int8_data21);
            __m128         data6           = _mm_cvtepi32_ps(cur_int8_data30);
            __m128         data7           = _mm_cvtepi32_ps(cur_int8_data31);

            data0 = _mm_sub_ps(data0, mean0);
            data1 = _mm_sub_ps(data1, mean0);
            data2 = _mm_sub_ps(data2, mean0);
			data3 = _mm_sub_ps(data3, mean0);
            data4 = _mm_sub_ps(data4, mean0);
            data5 = _mm_sub_ps(data5, mean0);
            data6 = _mm_sub_ps(data6, mean0);
            data7 = _mm_sub_ps(data7, mean0);


            data0 = _mm_mul_ps(data0, inv_std0);
            data1 = _mm_mul_ps(data1, inv_std0);
            data2 = _mm_mul_ps(data2, inv_std0);
            data3 = _mm_mul_ps(data3, inv_std0);
            data4 = _mm_mul_ps(data4, inv_std0);
            data5 = _mm_mul_ps(data5, inv_std0);
            data6 = _mm_mul_ps(data6, inv_std0);
            data7 = _mm_mul_ps(data7, inv_std0);

            _mm_storeu_ps(dst_ptr,      data0);
            _mm_storeu_ps(dst_ptr + 4,  data1);
            _mm_storeu_ps(dst_ptr + 8,  data2);
            _mm_storeu_ps(dst_ptr + 12, data3);
            _mm_storeu_ps(dst_ptr + 16, data4);
            _mm_storeu_ps(dst_ptr + 20, data5);
            _mm_storeu_ps(dst_ptr + 24, data6);
            _mm_storeu_ps(dst_ptr + 28, data7);

            src_ptr += 32;
            dst_ptr += 32;
        }
    }
}

void normalize_image_f1f1_implement_sse(const float*   src, 
                                        float*         dst, 
                                        int            width_aligned, 
                                        int            height, 
                                        int            src_line_element, 
                                        int            dst_line_element, 
                                        float          mean, 
                                        float          inv_std)
{
    int           i           = 0;
    int           j           = 0;
    __m128        mean0       = _mm_set1_ps(mean);
    __m128        inv_std0    = _mm_set1_ps(inv_std);
    for (i = 0; i < height; i++)
    {
		float const*  src_ptr = src + i * src_line_element;
		float*        dst_ptr = dst + i * dst_line_element;
		for (j = 0; j < width_aligned; j += 16)
		{
			__m128 data0 = _mm_loadu_ps(src_ptr);
			__m128 data1 = _mm_loadu_ps(src_ptr + 4);
			__m128 data2 = _mm_loadu_ps(src_ptr + 8);
			__m128 data3 = _mm_loadu_ps(src_ptr + 12);

			data0 = _mm_sub_ps(data0, mean0);
			data1 = _mm_sub_ps(data1, mean0);
			data2 = _mm_sub_ps(data2, mean0);
			data3 = _mm_sub_ps(data3, mean0);

			data0 = _mm_mul_ps(data0, inv_std0);
			data1 = _mm_mul_ps(data1, inv_std0);
			data2 = _mm_mul_ps(data2, inv_std0);
			data3 = _mm_mul_ps(data3, inv_std0);

			_mm_storeu_ps(dst_ptr,      data0);
			_mm_storeu_ps(dst_ptr + 4,  data1);
			_mm_storeu_ps(dst_ptr + 8,  data2);
			_mm_storeu_ps(dst_ptr + 12, data3);

			src_ptr += 16;
			dst_ptr += 16;
		}
	}
}

void normalize_image_f3f3_implement_sse(const float*   src, 
                                        float*         dst, 
                                        int            width_aligned, 
                                        int            height, 
                                        int            src_line_element, 
                                        int            dst_line_element, 
                                        const float*   mean, 
                                        const float*   inv_std)
{
	int           i                = 0;
	int           j                = 0;
    int           linesize_aligned = width_aligned * 3;
	__m128        mean0            = _mm_loadu_ps(mean);
	__m128        mean1            = _mm_loadu_ps(mean + 4);
	__m128        mean2            = _mm_loadu_ps(mean + 8);
	__m128        inv_std0         = _mm_loadu_ps(inv_std);
	__m128        inv_std1         = _mm_loadu_ps(inv_std + 4);
	__m128        inv_std2         = _mm_loadu_ps(inv_std + 8);
	for (i = 0; i < height; i++)
	{
		float const*  src_ptr  = src + i * src_line_element;
		float*        dst_ptr  = dst + i * dst_line_element;
		for (j = 0; j < linesize_aligned; j += 24)
		{
			__m128 data0 = _mm_loadu_ps(src_ptr);
			__m128 data1 = _mm_loadu_ps(src_ptr + 4);
			__m128 data2 = _mm_loadu_ps(src_ptr + 8);
			__m128 data3 = _mm_loadu_ps(src_ptr + 12);
			__m128 data4 = _mm_loadu_ps(src_ptr + 16);
			__m128 data5 = _mm_loadu_ps(src_ptr + 20);

			data0 = _mm_sub_ps(data0, mean0);
			data1 = _mm_sub_ps(data1, mean1);
			data2 = _mm_sub_ps(data2, mean2);
			data3 = _mm_sub_ps(data3, mean0);
			data4 = _mm_sub_ps(data4, mean1);
			data5 = _mm_sub_ps(data5, mean2);

			data0 = _mm_mul_ps(data0, inv_std0);
			data1 = _mm_mul_ps(data1, inv_std1);
			data2 = _mm_mul_ps(data2, inv_std2);
			data3 = _mm_mul_ps(data3, inv_std0);
			data4 = _mm_mul_ps(data4, inv_std1);
			data5 = _mm_mul_ps(data5, inv_std2);

			_mm_storeu_ps(dst_ptr,      data0);
			_mm_storeu_ps(dst_ptr + 4,  data1);
			_mm_storeu_ps(dst_ptr + 8,  data2);
			_mm_storeu_ps(dst_ptr + 12, data3);
			_mm_storeu_ps(dst_ptr + 16, data4);
			_mm_storeu_ps(dst_ptr + 20, data5);

			src_ptr += 24;
			dst_ptr += 24;
		}
	}
}

void normalize_image_f4f4_implement_sse(const float*    src, 
                                        float*          dst, 
                                        int             width_aligned, 
                                        int             height, 
                                        int             src_line_element, 
                                        int             dst_line_element, 
                                        const float*    mean, 
                                        const float*    inv_std)
{
    int           i                = 0;
    int           j                = 0;
    int           linesize_aligned = width_aligned * 4;
    __m128        mean0            = _mm_loadu_ps(mean);
    __m128        inv_std0         = _mm_loadu_ps(inv_std);
    for (i = 0; i < height; i++)
    {
        float const*  src_ptr  = src + i * src_line_element;
        float*        dst_ptr  = dst + i * dst_line_element;
        for (j = 0; j < linesize_aligned; j += 32)
        {
            __m128 data0 = _mm_loadu_ps(src_ptr);
            __m128 data1 = _mm_loadu_ps(src_ptr + 4);
            __m128 data2 = _mm_loadu_ps(src_ptr + 8);
            __m128 data3 = _mm_loadu_ps(src_ptr + 12);
            __m128 data4 = _mm_loadu_ps(src_ptr + 16);
            __m128 data5 = _mm_loadu_ps(src_ptr + 20);
            __m128 data6 = _mm_loadu_ps(src_ptr + 24);
            __m128 data7 = _mm_loadu_ps(src_ptr + 28);

            data0 = _mm_sub_ps(data0, mean0);
            data1 = _mm_sub_ps(data1, mean0);
            data2 = _mm_sub_ps(data2, mean0);
			data3 = _mm_sub_ps(data3, mean0);
            data4 = _mm_sub_ps(data4, mean0);
            data5 = _mm_sub_ps(data5, mean0);
            data6 = _mm_sub_ps(data6, mean0);
            data7 = _mm_sub_ps(data7, mean0);


            data0 = _mm_mul_ps(data0, inv_std0);
            data1 = _mm_mul_ps(data1, inv_std0);
            data2 = _mm_mul_ps(data2, inv_std0);
            data3 = _mm_mul_ps(data3, inv_std0);
            data4 = _mm_mul_ps(data4, inv_std0);
            data5 = _mm_mul_ps(data5, inv_std0);
            data6 = _mm_mul_ps(data6, inv_std0);
            data7 = _mm_mul_ps(data7, inv_std0);

            _mm_storeu_ps(dst_ptr,      data0);
            _mm_storeu_ps(dst_ptr + 4,  data1);
            _mm_storeu_ps(dst_ptr + 8,  data2);
            _mm_storeu_ps(dst_ptr + 12, data3);
            _mm_storeu_ps(dst_ptr + 16, data4);
            _mm_storeu_ps(dst_ptr + 20, data5);
            _mm_storeu_ps(dst_ptr + 24, data6);
            _mm_storeu_ps(dst_ptr + 28, data7);

            src_ptr += 32;
            dst_ptr += 32;
        }
    }
}