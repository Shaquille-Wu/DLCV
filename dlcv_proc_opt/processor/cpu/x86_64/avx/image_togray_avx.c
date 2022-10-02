#include <immintrin.h>

/*
void image_togray_uc3_implement_avx(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef)
{
    int i = 0, j = 0, k = 0;
    const __m256i  coef0  = _mm256_set1_epi16(cvt_coef[0]);
    const __m256i  coef1  = _mm256_set1_epi16(cvt_coef[1]);
    const __m256i  coef2  = _mm256_set1_epi16(cvt_coef[2]);
    const __m256i  half   = _mm256_set1_epi32(16384);

    const __m128i  m0     = _mm_setr_epi8(0,  0, -1,  0,  0, -1, 0,  0, -1,  0,  0, -1, 0,  0, -1,  0);
    const __m128i  m1     = _mm_setr_epi8(0, -1,  0,  0, -1,  0, 0, -1,  0,  0, -1,  0, 0, -1,  0,  0);
    const __m128i  sh_b   = _mm_setr_epi8(0,  3,  6,  9, 12, 15, 2,  5,  8, 11, 14,  1, 4,  7, 10, 13);
    const __m128i  sh_g   = _mm_setr_epi8(1,  4,  7, 10, 13,  0, 3,  6,  9, 12, 15,  2, 5,  8, 11, 14);
    const __m128i  sh_r   = _mm_setr_epi8(2,  5,  8, 11, 14,  1, 4,  7, 10, 13,  0,  3, 6,  9, 12, 15);
    const __m128i  zero   = _mm_set1_epi32(0);
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src;
        unsigned char*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 16)
        {
            __m256i data_0, data_1, data_2;
            {
                __m128i data00, data01, data02, data10, data11, data12;
                __m128i d0     = _mm_loadu_si128((const __m128i*)cur_src_ptr);
                __m128i d1     = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 16));
                __m128i d2     = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 32));
                __m128i data0  = _mm_blendv_epi8(_mm_blendv_epi8(d0, d1, m0), d2, m1);
                __m128i data1  = _mm_blendv_epi8(_mm_blendv_epi8(d1, d2, m0), d0, m1);
                __m128i data2  = _mm_blendv_epi8(_mm_blendv_epi8(d2, d0, m0), d1, m1);

                data0          = _mm_shuffle_epi8(data0, sh_b);
                data1          = _mm_shuffle_epi8(data1, sh_g);
                data2          = _mm_shuffle_epi8(data2, sh_r);

                data00         = _mm_unpacklo_epi8(data0, zero);
                data10         = _mm_unpackhi_epi8(data0, zero);
                data01         = _mm_unpacklo_epi8(data1, zero);
                data11         = _mm_unpackhi_epi8(data1, zero);
                data02         = _mm_unpacklo_epi8(data2, zero);
                data12         = _mm_unpackhi_epi8(data2, zero);

                data_0         = _mm256_inserti128_si256(data_0, data00, 0);
                data_0         = _mm256_inserti128_si256(data_0, data10, 1);
                data_1         = _mm256_inserti128_si256(data_1, data01, 0);
                data_1         = _mm256_inserti128_si256(data_1, data11, 1);
                data_2         = _mm256_inserti128_si256(data_2, data02, 0);
                data_2         = _mm256_inserti128_si256(data_2, data12, 1);
            }

            __m256i   mul0lo  = _mm256_mullo_epi16(data_0, coef0);
            __m256i   mul0hi  = _mm256_mulhi_epi16(data_0, coef0);
            __m256i   mul1lo  = _mm256_mullo_epi16(data_1, coef1);
            __m256i   mul1hi  = _mm256_mulhi_epi16(data_1, coef1);
            __m256i   mul2lo  = _mm256_mullo_epi16(data_2, coef2);
            __m256i   mul2hi  = _mm256_mulhi_epi16(data_2, coef2);

            __m256i   res000  = _mm256_unpacklo_epi16(mul0lo, mul0hi);
            __m256i   res001  = _mm256_unpackhi_epi16(mul0lo, mul0hi);
            __m256i   res010  = _mm256_unpacklo_epi16(mul1lo, mul1hi);
            __m256i   res011  = _mm256_unpackhi_epi16(mul1lo, mul1hi);
            __m256i   res020  = _mm256_unpacklo_epi16(mul2lo, mul2hi);
            __m256i   res021  = _mm256_unpackhi_epi16(mul2lo, mul2hi);

            res000 = _mm256_add_epi32(res000, res010);
            res001 = _mm256_add_epi32(res001, res011);
            res000 = _mm256_add_epi32(res000, res020);
            res001 = _mm256_add_epi32(res001, res021);
            res000 = _mm256_add_epi32(res000, half);
            res001 = _mm256_add_epi32(res001, half);
            res000 = _mm256_srli_epi32(res000, 15);
            res001 = _mm256_srli_epi32(res001, 15);
            res000 = _mm256_packus_epi32(res000, res001);
            res001 = _mm256_permute2x128_si256(res000, res000, 1);
            res000 = _mm256_packus_epi16(res000, res001);
            __m128i res = _mm256_extracti128_si256(res000, 0);
            _mm_storeu_si128((__m128i*)(cur_dst_ptr), res);

            cur_src_ptr += 48;
            cur_dst_ptr += 16;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}
*/

void image_togray_uc4_implement_avx(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef)
{
    int i = 0, j = 0, k = 0;
    const unsigned int cb = cvt_coef[0];
    const unsigned int cg = cvt_coef[1];
    const unsigned int cr = cvt_coef[2];
    const __m256i  coef0  = _mm256_set1_epi32(cb | (cr << 16));
    const __m256i  coef1  = _mm256_set1_epi32(cg);
    const __m256i  mask   = _mm256_set1_epi32(0x00FF00FF);
    const __m256i  half   = _mm256_set1_epi32(16384);
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src;
        unsigned char*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 16)
        {
            __m256i   data0   = _mm256_loadu_si256((const __m256i*)cur_src_ptr);        
            __m256i   data1   = _mm256_loadu_si256((const __m256i*)(cur_src_ptr + 32)); 
            __m256i   data00  = _mm256_and_si256(data0, mask);  //br
            __m256i   data01  = _mm256_srli_epi16(data0, 8);    //ga
            __m256i   data10  = _mm256_and_si256(data1, mask);  //br
            __m256i   data11  = _mm256_srli_epi16(data1, 8);    //ga

            __m256i   res00   = _mm256_madd_epi16(data00, coef0);           //br
            __m256i   res01   = _mm256_madd_epi16(data01, coef1);           //ga
            __m256i   res10   = _mm256_madd_epi16(data10, coef0);           //br
            __m256i   res11   = _mm256_madd_epi16(data11, coef1);           //ga

            res00             = _mm256_add_epi32(res00, res01);
            res10             = _mm256_add_epi32(res10, res11);
            res00             = _mm256_add_epi32(res00, half);
            res10             = _mm256_add_epi32(res10, half);
            res00             = _mm256_srli_epi32(res00, 15);
            res10             = _mm256_srli_epi32(res10, 15);
            res01             = _mm256_permute2x128_si256(res00, res10, 32);
            res11             = _mm256_permute2x128_si256(res00, res10, 49);
            res01             = _mm256_packus_epi32(res01, res11);
            res00             = _mm256_permute2x128_si256(res01, res01, 1);
            res01             = _mm256_packus_epi16(res01, res00);
            __m128i  res      = _mm256_extracti128_si256(res01, 0);
            _mm_storeu_si128((__m128i*)(cur_dst_ptr), res);

            cur_src_ptr += 64;
            cur_dst_ptr += 16;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}

void image_togray_f3_implement_avx(float const*   src,
                                   float*         dst,
                                   int            src_width,
                                   int            src_height,
                                   int            src_line_size,
                                   int            dst_line_size,
                                   float const*   cvt_coef)
{
    int i = 0, j = 0;
    const float cb = cvt_coef[0];
    const float cg = cvt_coef[1];
    const float cr = cvt_coef[2];
    const __m256  coef0  = _mm256_set1_ps(cb);
    const __m256  coef1  = _mm256_set1_ps(cg);
    const __m256  coef2  = _mm256_set1_ps(cr);
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src;
        float*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 8)
        {
            __m256  data0, data1, data2;
            {
                __m256i bgr0     = _mm256_loadu_si256((const __m256i*)(cur_src_ptr));
                __m256i bgr1     = _mm256_loadu_si256((const __m256i*)(cur_src_ptr + 8));
                __m256i bgr2     = _mm256_loadu_si256((const __m256i*)(cur_src_ptr + 16));
                __m256i s02_low  = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2*16);
                __m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3*16);
                __m256i b0       = _mm256_blend_epi32(_mm256_blend_epi32(s02_low,  s02_high, 0x24),     bgr1, 0x92);
                __m256i g0       = _mm256_blend_epi32(_mm256_blend_epi32(s02_high, s02_low,  0x92),     bgr1, 0x24);
                __m256i r0       = _mm256_blend_epi32(_mm256_blend_epi32(bgr1,     s02_low,  0x24), s02_high, 0x92);
                data0            = _mm256_castsi256_ps(_mm256_shuffle_epi32(b0, 0x6c));
                data1            = _mm256_castsi256_ps(_mm256_shuffle_epi32(g0, 0xb1));
                data2            = _mm256_castsi256_ps(_mm256_shuffle_epi32(r0, 0xc6));
            }

            data0 = _mm256_mul_ps(data0, coef0);
            data1 = _mm256_mul_ps(data1, coef1);
            data2 = _mm256_mul_ps(data2, coef2);
            data0 = _mm256_add_ps(data1, data0);
            data0 = _mm256_add_ps(data2, data0);

            _mm256_storeu_ps(cur_dst_ptr, data0);

            cur_src_ptr += 24;
            cur_dst_ptr += 8;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}

void image_togray_f4_implement_avx(float const*   src,
                                   float*         dst,
                                   int            src_width,
                                   int            src_height,
                                   int            src_line_size,
                                   int            dst_line_size,
                                   float const*   cvt_coef)
{
    int i = 0, j = 0;
    const float cb = cvt_coef[0];
    const float cg = cvt_coef[1];
    const float cr = cvt_coef[2];
    const __m256  coef0  = _mm256_set1_ps(cb);
    const __m256  coef1  = _mm256_set1_ps(cg);
    const __m256  coef2  = _mm256_set1_ps(cr);
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src;
        float*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 8)
        {
            __m256i   int_data_raw0   = _mm256_loadu_si256((__m256i*)cur_src_ptr);
            __m256i   int_data_raw1   = _mm256_loadu_si256((__m256i*)(cur_src_ptr + 8));
            __m256i   int_data_raw2   = _mm256_loadu_si256((__m256i*)(cur_src_ptr + 16));
            __m256i   int_data_raw3   = _mm256_loadu_si256((__m256i*)(cur_src_ptr + 24));
            __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
            {
                cur_int8_data0   = _mm256_unpacklo_epi32(int_data_raw0, int_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
                cur_int8_data2   = _mm256_unpacklo_epi32(int_data_raw2, int_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
                cur_int8_data1   = _mm256_unpackhi_epi32(int_data_raw0, int_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
                cur_int8_data3   = _mm256_unpackhi_epi32(int_data_raw2, int_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
                int_data_raw0    = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
                int_data_raw1    = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
                int_data_raw2    = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
                int_data_raw3    = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
                cur_int8_data0   = _mm256_unpacklo_epi32(int_data_raw0, int_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
                cur_int8_data2   = _mm256_unpacklo_epi32(int_data_raw2, int_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
                cur_int8_data1   = _mm256_unpackhi_epi32(int_data_raw0, int_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
                cur_int8_data3   = _mm256_unpackhi_epi32(int_data_raw2, int_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
                int_data_raw0    = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
                int_data_raw2    = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
                int_data_raw1    = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            }
            __m256 data0 = _mm256_castsi256_ps(int_data_raw0);
            __m256 data1 = _mm256_castsi256_ps(int_data_raw1);
            __m256 data2 = _mm256_castsi256_ps(int_data_raw2);

            data0 = _mm256_mul_ps(data0, coef0);
            data1 = _mm256_mul_ps(data1, coef1);
            data2 = _mm256_mul_ps(data2, coef2);
            data0 = _mm256_add_ps(data1, data0);
            data0 = _mm256_add_ps(data2, data0);

            _mm256_storeu_ps(cur_dst_ptr, data0);

            cur_src_ptr += 32;
            cur_dst_ptr += 8;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}