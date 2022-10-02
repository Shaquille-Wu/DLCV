#include <immintrin.h>

void normalize_image_uc1f1_implement_avx(const unsigned char*  src, 
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
	__m256        mean0       = _mm256_set1_ps(mean);
	__m256        inv_std0    = _mm256_set1_ps(inv_std);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr = src + i * src_line_element;
        float*                dst_ptr = dst + i * dst_line_element;
        for (j = 0; j < width_aligned; j += 32)
        {
            __m128i   int8_data_raw0  = _mm_loadu_si128((__m128i*)(src_ptr));
            __m128i   int8_data_raw1  = _mm_loadu_si128((__m128i*)(src_ptr + 16));

            __m256i   cur_int8_data0  = _mm256_cvtepu8_epi32(int8_data_raw0);
            __m256i   cur_int8_data1  = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 8));
            __m256i   cur_int8_data2  = _mm256_cvtepu8_epi32(int8_data_raw1);
            __m256i   cur_int8_data3  = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw1, 8));

            __m256    data0           = _mm256_cvtepi32_ps(cur_int8_data0);
            __m256    data1           = _mm256_cvtepi32_ps(cur_int8_data1);
            __m256    data2           = _mm256_cvtepi32_ps(cur_int8_data2);
            __m256    data3           = _mm256_cvtepi32_ps(cur_int8_data3);

            data0 = _mm256_sub_ps(data0, mean0);
            data1 = _mm256_sub_ps(data1, mean0);
            data2 = _mm256_sub_ps(data2, mean0);
            data3 = _mm256_sub_ps(data3, mean0);

            data0 = _mm256_mul_ps(data0, inv_std0);
            data1 = _mm256_mul_ps(data1, inv_std0);
            data2 = _mm256_mul_ps(data2, inv_std0);
            data3 = _mm256_mul_ps(data3, inv_std0);

            _mm256_storeu_ps(dst_ptr,      data0);
            _mm256_storeu_ps(dst_ptr + 8,  data1);
            _mm256_storeu_ps(dst_ptr + 16, data2);
            _mm256_storeu_ps(dst_ptr + 24, data3);

            src_ptr += 32;
            dst_ptr += 32;
        }
    }
}

void normalize_image_uc3f3_implement_avx(const unsigned char*  src, 
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
	__m256        mean0            = _mm256_loadu_ps(mean);
	__m256        mean1            = _mm256_loadu_ps(mean + 8);
	__m256        mean2            = _mm256_loadu_ps(mean + 16);
	__m256        inv_std0         = _mm256_loadu_ps(inv_std);
	__m256        inv_std1         = _mm256_loadu_ps(inv_std + 8);
	__m256        inv_std2         = _mm256_loadu_ps(inv_std + 16);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr  = src + i * src_line_element;
        float*                dst_ptr  = dst + i * dst_line_element;
        for (j = 0; j < linesize_aligned; j += 48)
        {
            __m128i        int8_data_raw0  = _mm_loadu_si128((__m128i*)src_ptr);
			__m128i        int8_data_raw1  = _mm_loadu_si128((__m128i*)(src_ptr + 16));
			__m128i        int8_data_raw2  = _mm_loadu_si128((__m128i*)(src_ptr + 32));

            __m256i        cur_int8_data00 = _mm256_cvtepu8_epi32(int8_data_raw0);
            __m256i        cur_int8_data01 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 8));
            __m256i        cur_int8_data10 = _mm256_cvtepu8_epi32(int8_data_raw1);
            __m256i        cur_int8_data11 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw1, 8));
            __m256i        cur_int8_data20 = _mm256_cvtepu8_epi32(int8_data_raw2);
            __m256i        cur_int8_data21 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 8));

            __m256         data0           = _mm256_cvtepi32_ps(cur_int8_data00);
            __m256         data1           = _mm256_cvtepi32_ps(cur_int8_data01);
            __m256         data2           = _mm256_cvtepi32_ps(cur_int8_data10);
            __m256         data3           = _mm256_cvtepi32_ps(cur_int8_data11);
            __m256         data4           = _mm256_cvtepi32_ps(cur_int8_data20);
            __m256         data5           = _mm256_cvtepi32_ps(cur_int8_data21);

            data0 = _mm256_sub_ps(data0, mean0);
            data1 = _mm256_sub_ps(data1, mean1);
            data2 = _mm256_sub_ps(data2, mean2);
            data3 = _mm256_sub_ps(data3, mean0);
            data4 = _mm256_sub_ps(data4, mean1);
            data5 = _mm256_sub_ps(data5, mean2);

            data0 = _mm256_mul_ps(data0, inv_std0);
            data1 = _mm256_mul_ps(data1, inv_std1);
            data2 = _mm256_mul_ps(data2, inv_std2);
            data3 = _mm256_mul_ps(data3, inv_std0);
            data4 = _mm256_mul_ps(data4, inv_std1);
            data5 = _mm256_mul_ps(data5, inv_std2);

            _mm256_storeu_ps(dst_ptr,      data0);
            _mm256_storeu_ps(dst_ptr + 8,  data1);
            _mm256_storeu_ps(dst_ptr + 16, data2);
            _mm256_storeu_ps(dst_ptr + 24, data3);
            _mm256_storeu_ps(dst_ptr + 32, data4);
            _mm256_storeu_ps(dst_ptr + 40, data5);

            src_ptr += 48;
            dst_ptr += 48;
        }
    }
}

void normalize_image_uc4f4_implement_avx(const unsigned char*  src, 
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
	__m256        mean0            = _mm256_loadu_ps(mean);
	__m256        inv_std0         = _mm256_loadu_ps(inv_std);
    for (i = 0; i < height; i++)
    {
        unsigned char const*  src_ptr  = src + i * src_line_element;
        float*                dst_ptr  = dst + i * dst_line_element;
        for (j = 0; j < linesize_aligned; j += 64)
        {
            __m128i        int8_data_raw0  = _mm_loadu_si128((__m128i*)src_ptr);
			__m128i        int8_data_raw1  = _mm_loadu_si128((__m128i*)(src_ptr + 16));
			__m128i        int8_data_raw2  = _mm_loadu_si128((__m128i*)(src_ptr + 32));
			__m128i        int8_data_raw3  = _mm_loadu_si128((__m128i*)(src_ptr + 48));

            __m256i        cur_int8_data00 = _mm256_cvtepu8_epi32(int8_data_raw0);
            __m256i        cur_int8_data01 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw0, 8));
            __m256i        cur_int8_data10 = _mm256_cvtepu8_epi32(int8_data_raw1);
            __m256i        cur_int8_data11 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw1, 8));
            __m256i        cur_int8_data20 = _mm256_cvtepu8_epi32(int8_data_raw2);
            __m256i        cur_int8_data21 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw2, 8));
            __m256i        cur_int8_data30 = _mm256_cvtepu8_epi32(int8_data_raw3);
            __m256i        cur_int8_data31 = _mm256_cvtepu8_epi32(_mm_srli_si128(int8_data_raw3, 8));

            __m256         data0           = _mm256_cvtepi32_ps(cur_int8_data00);
            __m256         data1           = _mm256_cvtepi32_ps(cur_int8_data01);
            __m256         data2           = _mm256_cvtepi32_ps(cur_int8_data10);
            __m256         data3           = _mm256_cvtepi32_ps(cur_int8_data11);
            __m256         data4           = _mm256_cvtepi32_ps(cur_int8_data20);
            __m256         data5           = _mm256_cvtepi32_ps(cur_int8_data21);
            __m256         data6           = _mm256_cvtepi32_ps(cur_int8_data30);
            __m256         data7           = _mm256_cvtepi32_ps(cur_int8_data31);

            data0 = _mm256_sub_ps(data0, mean0);
            data1 = _mm256_sub_ps(data1, mean0);
            data2 = _mm256_sub_ps(data2, mean0);
            data3 = _mm256_sub_ps(data3, mean0);
            data4 = _mm256_sub_ps(data4, mean0);
            data5 = _mm256_sub_ps(data5, mean0);
            data6 = _mm256_sub_ps(data6, mean0);
            data7 = _mm256_sub_ps(data7, mean0);

            data0 = _mm256_mul_ps(data0, inv_std0);
            data1 = _mm256_mul_ps(data1, inv_std0);
            data2 = _mm256_mul_ps(data2, inv_std0);
            data3 = _mm256_mul_ps(data3, inv_std0);
            data4 = _mm256_mul_ps(data4, inv_std0);
            data5 = _mm256_mul_ps(data5, inv_std0);
            data6 = _mm256_mul_ps(data6, inv_std0);
            data7 = _mm256_mul_ps(data7, inv_std0);

            _mm256_storeu_ps(dst_ptr,      data0);
            _mm256_storeu_ps(dst_ptr + 8,  data1);
            _mm256_storeu_ps(dst_ptr + 16, data2);
            _mm256_storeu_ps(dst_ptr + 24, data3);
            _mm256_storeu_ps(dst_ptr + 32, data4);
            _mm256_storeu_ps(dst_ptr + 40, data5);
            _mm256_storeu_ps(dst_ptr + 48, data6);
            _mm256_storeu_ps(dst_ptr + 56, data7);

            src_ptr += 64;
            dst_ptr += 64;
        }
    }
}

void normalize_image_f1f1_implement_avx(const float* src, float* dst, int width_aligned, int height, int src_line_element, int dst_line_element, float mean, float inv_std)
{
	int           i           = 0;
	int           j           = 0;
	__m256        mean0       = _mm256_set1_ps(mean);
	__m256        inv_std0    = _mm256_set1_ps(inv_std);
	for (i = 0; i < height; i++)
	{
		float const*  src_ptr = src + i * src_line_element;
		float*        dst_ptr = dst + i * dst_line_element;
		for (j = 0; j < width_aligned; j += 32)
		{
			__m256 data0 = _mm256_loadu_ps(src_ptr);
			__m256 data1 = _mm256_loadu_ps(src_ptr + 8);
			__m256 data2 = _mm256_loadu_ps(src_ptr + 16);
			__m256 data3 = _mm256_loadu_ps(src_ptr + 24);

			data0 = _mm256_sub_ps(data0, mean0);
			data1 = _mm256_sub_ps(data1, mean0);
			data2 = _mm256_sub_ps(data2, mean0);
			data3 = _mm256_sub_ps(data3, mean0);

			data0 = _mm256_mul_ps(data0, inv_std0);
			data1 = _mm256_mul_ps(data1, inv_std0);
			data2 = _mm256_mul_ps(data2, inv_std0);
			data3 = _mm256_mul_ps(data3, inv_std0);

			_mm256_storeu_ps(dst_ptr,      data0);
			_mm256_storeu_ps(dst_ptr + 8,  data1);
			_mm256_storeu_ps(dst_ptr + 16, data2);
			_mm256_storeu_ps(dst_ptr + 24, data3);

			src_ptr += 32;
			dst_ptr += 32;
		}
	}
}

void normalize_image_f3f3_implement_avx(const float* src, float* dst, int width_aligned, int height, int src_line_element, int dst_line_element, const float* mean, const float* inv_std)
{
	int           i                = 0;
	int           j                = 0;
    int           linesize_alinged = 3 * width_aligned;
	__m256        mean0            = _mm256_loadu_ps(mean);
	__m256        mean1            = _mm256_loadu_ps(mean + 8);
	__m256        mean2            = _mm256_loadu_ps(mean + 16);
	__m256        inv_std0         = _mm256_loadu_ps(inv_std);
	__m256        inv_std1         = _mm256_loadu_ps(inv_std + 8);
	__m256        inv_std2         = _mm256_loadu_ps(inv_std + 16);
	for (i = 0; i < height; i++)
	{
		float const*  src_ptr = src + i * src_line_element;
		float*        dst_ptr = dst + i * dst_line_element;
		for (j = 0; j < linesize_alinged; j += 48)
		{
			__m256 data0 = _mm256_loadu_ps(src_ptr);
			__m256 data1 = _mm256_loadu_ps(src_ptr + 8);
			__m256 data2 = _mm256_loadu_ps(src_ptr + 16);
			__m256 data3 = _mm256_loadu_ps(src_ptr + 24);
			__m256 data4 = _mm256_loadu_ps(src_ptr + 32);
			__m256 data5 = _mm256_loadu_ps(src_ptr + 40);

			data0 = _mm256_sub_ps(data0, mean0);
			data1 = _mm256_sub_ps(data1, mean1);
			data2 = _mm256_sub_ps(data2, mean2);
			data3 = _mm256_sub_ps(data3, mean0);
			data4 = _mm256_sub_ps(data4, mean1);
			data5 = _mm256_sub_ps(data5, mean2);

			data0 = _mm256_mul_ps(data0, inv_std0);
			data1 = _mm256_mul_ps(data1, inv_std1);
			data2 = _mm256_mul_ps(data2, inv_std2);
			data3 = _mm256_mul_ps(data3, inv_std0);
			data4 = _mm256_mul_ps(data4, inv_std1);
			data5 = _mm256_mul_ps(data5, inv_std2);

			_mm256_storeu_ps(dst_ptr,      data0);
			_mm256_storeu_ps(dst_ptr + 8,  data1);
			_mm256_storeu_ps(dst_ptr + 16, data2);
			_mm256_storeu_ps(dst_ptr + 24, data3);
			_mm256_storeu_ps(dst_ptr + 32, data4);
			_mm256_storeu_ps(dst_ptr + 40, data5);

			src_ptr += 48;
			dst_ptr += 48;
		}
	}
}


void normalize_image_f4f4_implement_avx(const float* src, float* dst, int width_aligned, int height, int src_line_element, int dst_line_element, const float* mean, const float* inv_std)
{
	int           i                = 0;
	int           j                = 0;
    int           linesize_alinged = 4 * width_aligned;
	__m256        mean0            = _mm256_loadu_ps(mean);
	__m256        inv_std0         = _mm256_loadu_ps(inv_std);
	for (i = 0; i < height; i++)
	{
		float const*  src_ptr = src + i * src_line_element;
		float*        dst_ptr = dst + i * dst_line_element;
		for (j = 0; j < linesize_alinged; j += 64)
		{
			__m256 data0 = _mm256_loadu_ps(src_ptr);
			__m256 data1 = _mm256_loadu_ps(src_ptr + 8);
			__m256 data2 = _mm256_loadu_ps(src_ptr + 16);
			__m256 data3 = _mm256_loadu_ps(src_ptr + 24);
			__m256 data4 = _mm256_loadu_ps(src_ptr + 32);
			__m256 data5 = _mm256_loadu_ps(src_ptr + 40);
			__m256 data6 = _mm256_loadu_ps(src_ptr + 48);
			__m256 data7 = _mm256_loadu_ps(src_ptr + 56);

			data0 = _mm256_sub_ps(data0, mean0);
			data1 = _mm256_sub_ps(data1, mean0);
			data2 = _mm256_sub_ps(data2, mean0);
			data3 = _mm256_sub_ps(data3, mean0);
			data4 = _mm256_sub_ps(data4, mean0);
			data5 = _mm256_sub_ps(data5, mean0);
			data6 = _mm256_sub_ps(data6, mean0);
			data7 = _mm256_sub_ps(data7, mean0);

			data0 = _mm256_mul_ps(data0, inv_std0);
			data1 = _mm256_mul_ps(data1, inv_std0);
			data2 = _mm256_mul_ps(data2, inv_std0);
			data3 = _mm256_mul_ps(data3, inv_std0);
			data4 = _mm256_mul_ps(data4, inv_std0);
			data5 = _mm256_mul_ps(data5, inv_std0);
			data6 = _mm256_mul_ps(data6, inv_std0);
			data7 = _mm256_mul_ps(data7, inv_std0);

			_mm256_storeu_ps(dst_ptr,      data0);
			_mm256_storeu_ps(dst_ptr + 8,  data1);
			_mm256_storeu_ps(dst_ptr + 16, data2);
			_mm256_storeu_ps(dst_ptr + 24, data3);
			_mm256_storeu_ps(dst_ptr + 32, data4);
			_mm256_storeu_ps(dst_ptr + 40, data5);
			_mm256_storeu_ps(dst_ptr + 48, data6);
			_mm256_storeu_ps(dst_ptr + 56, data7);

			src_ptr += 64;
			dst_ptr += 64;
		}
	}
}