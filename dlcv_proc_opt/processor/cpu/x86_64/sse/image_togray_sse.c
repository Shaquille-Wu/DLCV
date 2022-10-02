#include <emmintrin.h>
#include <smmintrin.h>

#if 0
void image_togray_uc3_implement_sse(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef)
{
    int i = 0, j = 0, k = 0;
    const __m128i  coef0  = _mm_set1_epi16(cvt_coef[0]);
    const __m128i  coef1  = _mm_set1_epi16(cvt_coef[1]);
    const __m128i  coef2  = _mm_set1_epi16(cvt_coef[2]);
    const __m128i  zero   = _mm_set1_epi32(0);
    const __m128i  half   = _mm_set1_epi32(16384);
    const __m128i  m0     = _mm_setr_epi8(0,  0, -1,  0,  0, -1, 0,  0, -1,  0,  0, -1, 0,  0, -1,  0);
    const __m128i  m1     = _mm_setr_epi8(0, -1,  0,  0, -1,  0, 0, -1,  0,  0, -1,  0, 0, -1,  0,  0);
    const __m128i  sh_b   = _mm_setr_epi8(0,  3,  6,  9, 12, 15, 2,  5,  8, 11, 14,  1, 4,  7, 10, 13);
    const __m128i  sh_g   = _mm_setr_epi8(1,  4,  7, 10, 13,  0, 3,  6,  9, 12, 15,  2, 5,  8, 11, 14);
    const __m128i  sh_r   = _mm_setr_epi8(2,  5,  8, 11, 14,  1, 4,  7, 10, 13,  0,  3, 6,  9, 12, 15);
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src;
        unsigned char*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 16)
        {
            __m128i data00, data01, data02, data10, data11, data12;
            {
                __m128i d0     = _mm_loadu_si128((const __m128i*)cur_src_ptr);
                __m128i d1     = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 16));
                __m128i d2     = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 32));
                
                __m128i data0  = _mm_blendv_epi8(_mm_blendv_epi8(d0, d1, m0), d2, m1);
                __m128i data1  = _mm_blendv_epi8(_mm_blendv_epi8(d1, d2, m0), d0, m1);
                __m128i data2  = _mm_blendv_epi8(_mm_blendv_epi8(d2, d0, m0), d1, m1);

                data0          = _mm_shuffle_epi8(data0, sh_b);
                data1          = _mm_shuffle_epi8(data1, sh_g);
                data2          = _mm_shuffle_epi8(data2, sh_r);

                data00        = _mm_unpacklo_epi8(data0, zero);
                data10        = _mm_unpackhi_epi8(data0, zero);
                data01        = _mm_unpacklo_epi8(data1, zero);
                data11        = _mm_unpackhi_epi8(data1, zero);
                data02        = _mm_unpacklo_epi8(data2, zero);
                data12        = _mm_unpackhi_epi8(data2, zero);
            }

            __m128i   mul0lo  = _mm_mullo_epi16(data00, coef0);
            __m128i   mul0hi  = _mm_mulhi_epi16(data00, coef0);
            __m128i   mul1lo  = _mm_mullo_epi16(data01, coef1);
            __m128i   mul1hi  = _mm_mulhi_epi16(data01, coef1);
            __m128i   mul2lo  = _mm_mullo_epi16(data02, coef2);
            __m128i   mul2hi  = _mm_mulhi_epi16(data02, coef2);

            __m128i   res000  = _mm_unpacklo_epi16(mul0lo, mul0hi);
            __m128i   res001  = _mm_unpackhi_epi16(mul0lo, mul0hi);
            __m128i   res010  = _mm_unpacklo_epi16(mul1lo, mul1hi);
            __m128i   res011  = _mm_unpackhi_epi16(mul1lo, mul1hi);
            __m128i   res020  = _mm_unpacklo_epi16(mul2lo, mul2hi);
            __m128i   res021  = _mm_unpackhi_epi16(mul2lo, mul2hi);
            res000 = _mm_add_epi32(res000, res010);
            res001 = _mm_add_epi32(res001, res011);
            res000 = _mm_add_epi32(res000, res020);
            res001 = _mm_add_epi32(res001, res021);
            res000 = _mm_add_epi32(res000, half);
            res001 = _mm_add_epi32(res001, half);
            res000 = _mm_srli_epi32(res000, 15);
            res001 = _mm_srli_epi32(res001, 15);
            res000 = _mm_packus_epi32(res000, res001);

            mul0lo = _mm_mullo_epi16(data10, coef0);
            mul0hi = _mm_mulhi_epi16(data10, coef0);
            mul1lo = _mm_mullo_epi16(data11, coef1);
            mul1hi = _mm_mulhi_epi16(data11, coef1);
            mul2lo = _mm_mullo_epi16(data12, coef2);
            mul2hi = _mm_mulhi_epi16(data12, coef2);

            __m128i   res100 = _mm_unpacklo_epi16(mul0lo, mul0hi);
            res001  = _mm_unpackhi_epi16(mul0lo, mul0hi);
            res010  = _mm_unpacklo_epi16(mul1lo, mul1hi);
            res011  = _mm_unpackhi_epi16(mul1lo, mul1hi);
            res020  = _mm_unpacklo_epi16(mul2lo, mul2hi);
            res021  = _mm_unpackhi_epi16(mul2lo, mul2hi);
            res100 = _mm_add_epi32(res100, res010);
            res001 = _mm_add_epi32(res001, res011);
            res100 = _mm_add_epi32(res100, res020);
            res001 = _mm_add_epi32(res001, res021);
            res100 = _mm_add_epi32(res100, half);
            res001 = _mm_add_epi32(res001, half);
            res100 = _mm_srli_epi32(res100, 15);
            res001 = _mm_srli_epi32(res001, 15);
            res100 = _mm_packus_epi32(res100, res001);

            res000 = _mm_packus_epi16(res000, res100);
            _mm_storeu_si128((__m128i*)(cur_dst_ptr), res000);

            cur_src_ptr += 48;
            cur_dst_ptr += 16;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}
#endif

void image_togray_uc3_implement_sse(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef)
{
    int i = 0, j = 0;
    const unsigned int cb = cvt_coef[0];
    const unsigned int cg = cvt_coef[1];
    const unsigned int cr = cvt_coef[2];
    const __m128i  coef0       = _mm_set1_epi32(cb | (cg << 16));
    const __m128i  coef1       = _mm_set1_epi32(cr);
    const __m128i  half        = _mm_set1_epi32(16384);
    const __m128i  s_mask0     = _mm_setr_epi8( 0, -1,  1, -1,  3, -1,  4, -1,  6, -1,  7, -1,  9, -1, 10, -1);
    const __m128i  s_mask1     = _mm_setr_epi8( 2, -1, -1, -1,  5, -1, -1, -1,  8, -1, -1, -1, 11, -1, -1, -1);
    const __m128i  s_mask2     = _mm_setr_epi8(12, -1, 13, -1, 15, -1,  0, -1,  2, -1,  3, -1,  5, -1,  6, -1);
    const __m128i  s_mask3     = _mm_setr_epi8(14, -1, -1, -1,  1, -1, -1, -1,  4, -1, -1, -1,  7, -1, -1, -1);
    const __m128i  s_mask4     = _mm_setr_epi8( 8, -1,  9, -1, 11, -1, 12, -1, 14, -1, 15, -1,  1, -1,  2, -1);
    const __m128i  s_mask5     = _mm_setr_epi8(10, -1, -1, -1, 13, -1, -1, -1,  0, -1, -1, -1,  3, -1, -1, -1);
    const __m128i  s_mask6     = _mm_setr_epi8( 4, -1,  5, -1,  7, -1,  8, -1, 10, -1, 11, -1, 13, -1, 14, -1);
    const __m128i  s_mask7     = _mm_setr_epi8( 6, -1, -1, -1,  9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1);
    const __m128i  blend_mask0 = _mm_setr_epi32(0, 0,          0, 0xFFFFFFFF);
    const __m128i  blend_mask1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src;
        unsigned char*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 16)
        {
            __m128i data00, data01, data10, data11;
            __m128i data20, data21, data30, data31;
            {

                __m128i d0      = _mm_loadu_si128((const __m128i*)cur_src_ptr);
                __m128i d1      = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 16));
                __m128i d2      = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 32));
                __m128i blend_0 = _mm_blendv_epi8(d1, d0, blend_mask0);
                __m128i blend_1 = _mm_blendv_epi8(d2, d1, blend_mask1);
                data00          = _mm_shuffle_epi8(d0,      s_mask0);           //bg
                data01          = _mm_shuffle_epi8(d0,      s_mask1);           //ra
                data10          = _mm_shuffle_epi8(blend_0, s_mask2);           //bg
                data11          = _mm_shuffle_epi8(blend_0, s_mask3);           //ra
                data20          = _mm_shuffle_epi8(blend_1, s_mask4);           //bg
                data21          = _mm_shuffle_epi8(blend_1, s_mask5);           //ra
                data30          = _mm_shuffle_epi8(d2,      s_mask6);           //bg
                data31          = _mm_shuffle_epi8(d2,      s_mask7);           //ra
            }

            __m128i   res00   = _mm_madd_epi16(data00, coef0);           //bg
            __m128i   res01   = _mm_madd_epi16(data01, coef1);           //ra
            __m128i   res10   = _mm_madd_epi16(data10, coef0);           //bg
            __m128i   res11   = _mm_madd_epi16(data11, coef1);           //ra
            __m128i   res20   = _mm_madd_epi16(data20, coef0);           //bg
            __m128i   res21   = _mm_madd_epi16(data21, coef1);           //ra
            __m128i   res30   = _mm_madd_epi16(data30, coef0);           //bg
            __m128i   res31   = _mm_madd_epi16(data31, coef1);           //ra

            res00             = _mm_add_epi32(res00, res01);
            res10             = _mm_add_epi32(res10, res11);
            res20             = _mm_add_epi32(res20, res21);
            res30             = _mm_add_epi32(res30, res31);
            res00             = _mm_add_epi32(res00, half);
            res10             = _mm_add_epi32(res10, half);
            res20             = _mm_add_epi32(res20, half);
            res30             = _mm_add_epi32(res30, half);
            res00             = _mm_srli_epi32(res00, 15);
            res10             = _mm_srli_epi32(res10, 15);
            res20             = _mm_srli_epi32(res20, 15);
            res30             = _mm_srli_epi32(res30, 15);
            res00             = _mm_packus_epi32(res00, res10);
            res20             = _mm_packus_epi32(res20, res30);
            res00             = _mm_packus_epi16(res00, res20);
            _mm_storeu_si128((__m128i*)(cur_dst_ptr), res00);

            cur_src_ptr += 48;
            cur_dst_ptr += 16;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}

void image_togray_uc4_implement_sse(unsigned char const*        src,
                                    unsigned char*              dst,
                                    int                         src_width,
                                    int                         src_height,
                                    int                         src_line_size,
                                    int                         dst_line_size,
                                    unsigned short int const*   cvt_coef)
{
    int i = 0, j = 0;
    const unsigned int cb = cvt_coef[0];
    const unsigned int cg = cvt_coef[1];
    const unsigned int cr = cvt_coef[2];
    const __m128i  coef0  = _mm_set1_epi32(cb | (cr << 16));
    const __m128i  coef1  = _mm_set1_epi32(cg);
    const __m128i  mask   = _mm_set1_epi32(0x00FF00FF);
    const __m128i  half   = _mm_set1_epi32(16384);
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src;
        unsigned char*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 16)
        {
            __m128i   data0   = _mm_loadu_si128((const __m128i*)cur_src_ptr);        
            __m128i   data1   = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 16)); 
            __m128i   data2   = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 32));
            __m128i   data3   = _mm_loadu_si128((const __m128i*)(cur_src_ptr + 48)); 
            __m128i   data00  = _mm_and_si128(data0, mask);  //br
            __m128i   data01  = _mm_srli_epi16(data0, 8);    //ga
            __m128i   data10  = _mm_and_si128(data1, mask);  //br
            __m128i   data11  = _mm_srli_epi16(data1, 8);    //ga
            __m128i   data20  = _mm_and_si128(data2, mask);  //br
            __m128i   data21  = _mm_srli_epi16(data2, 8);    //ga
            __m128i   data30  = _mm_and_si128(data3, mask);  //br
            __m128i   data31  = _mm_srli_epi16(data3, 8);    //ga

            __m128i   res00   = _mm_madd_epi16(data00, coef0);           //br
            __m128i   res01   = _mm_madd_epi16(data01, coef1);           //ga
            __m128i   res10   = _mm_madd_epi16(data10, coef0);           //br
            __m128i   res11   = _mm_madd_epi16(data11, coef1);           //ga
            __m128i   res20   = _mm_madd_epi16(data20, coef0);           //br
            __m128i   res21   = _mm_madd_epi16(data21, coef1);           //ga
            __m128i   res30   = _mm_madd_epi16(data30, coef0);           //br
            __m128i   res31   = _mm_madd_epi16(data31, coef1);           //ga

            res00             = _mm_add_epi32(res00, res01);
            res10             = _mm_add_epi32(res10, res11);
            res20             = _mm_add_epi32(res20, res21);
            res30             = _mm_add_epi32(res30, res31);
            res00             = _mm_add_epi32(res00, half);
            res10             = _mm_add_epi32(res10, half);
            res20             = _mm_add_epi32(res20, half);
            res30             = _mm_add_epi32(res30, half);
            res00             = _mm_srli_epi32(res00, 15);
            res10             = _mm_srli_epi32(res10, 15);
            res20             = _mm_srli_epi32(res20, 15);
            res30             = _mm_srli_epi32(res30, 15);
            res00             = _mm_packus_epi32(res00, res10);
            res20             = _mm_packus_epi32(res20, res30);
            res00             = _mm_packus_epi16(res00, res20);
            _mm_storeu_si128((__m128i*)(cur_dst_ptr), res00);

            cur_src_ptr += 64;
            cur_dst_ptr += 16;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}

void image_togray_f3_implement_sse(float const*   src,
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
    const __m128  coef0  = _mm_set1_ps(cb);
    const __m128  coef1  = _mm_set1_ps(cg);
    const __m128  coef2  = _mm_set1_ps(cr);
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src;
        float*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 4)
        {
            __m128 d0    = _mm_loadu_ps(cur_src_ptr);        
            __m128 d1    = _mm_loadu_ps(cur_src_ptr + 4); 
            __m128 d2    = _mm_loadu_ps(cur_src_ptr + 8);
            __m128 at12  = _mm_shuffle_ps(d1,     d2, _MM_SHUFFLE(0, 1, 0, 2));
            __m128 data0 = _mm_shuffle_ps(d0,   at12, _MM_SHUFFLE(2, 0, 3, 0));
            __m128 bt01  = _mm_shuffle_ps(d0,     d1, _MM_SHUFFLE(0, 0, 0, 1));
            __m128 bt12  = _mm_shuffle_ps(d1,     d2, _MM_SHUFFLE(0, 2, 0, 3));
            __m128 data1 = _mm_shuffle_ps(bt01, bt12, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 ct01  = _mm_shuffle_ps(d0,     d1, _MM_SHUFFLE(0, 1, 0, 2));
            __m128 data2 = _mm_shuffle_ps(ct01,   d2, _MM_SHUFFLE(3, 0, 2, 0));

            data0 = _mm_mul_ps(data0, coef0);
            data1 = _mm_mul_ps(data1, coef1);
            data2 = _mm_mul_ps(data2, coef2);
            data0 = _mm_add_ps(data1, data0);
            data0 = _mm_add_ps(data2, data0);

            _mm_storeu_ps(cur_dst_ptr, data0);

            cur_src_ptr += 12;
            cur_dst_ptr += 4;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}

void image_togray_f4_implement_sse(float const*   src,
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
    const __m128  coef0  = _mm_set1_ps(cb);
    const __m128  coef1  = _mm_set1_ps(cg);
    const __m128  coef2  = _mm_set1_ps(cr);
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src;
        float*        cur_dst_ptr = dst;
        for(j = 0 ; j < src_width ; j += 4)
        {
            __m128   data0       = _mm_loadu_ps(cur_src_ptr);        
            __m128   data1       = _mm_loadu_ps(cur_src_ptr + 4); 
            __m128   data2       = _mm_loadu_ps(cur_src_ptr + 8);
            __m128   data3       = _mm_loadu_ps(cur_src_ptr + 12);
            __m128   cur_data0   = _mm_unpacklo_ps(data0, data1);
            __m128   cur_data2   = _mm_unpacklo_ps(data2, data3);
            __m128   cur_data1   = _mm_unpackhi_ps(data0, data1);
            __m128   cur_data3   = _mm_unpackhi_ps(data2, data3);
            data0   = _mm_movelh_ps(cur_data0, cur_data2);
            data1   = _mm_movehl_ps(cur_data2, cur_data0);
            data2   = _mm_movelh_ps(cur_data1, cur_data3);

            data0 = _mm_mul_ps(data0, coef0);
            data1 = _mm_mul_ps(data1, coef1);
            data2 = _mm_mul_ps(data2, coef2);
            data0 = _mm_add_ps(data1, data0);
            data0 = _mm_add_ps(data2, data0);

            _mm_storeu_ps(cur_dst_ptr, data0);

            cur_src_ptr += 16;
            cur_dst_ptr += 4;
        }
        src += src_line_size;
        dst += dst_line_size;
    }
}