#include <emmintrin.h>
#include <smmintrin.h>
#include "../../../../common/dlcv_proc_opt_com_def.h"

#ifdef RESIZE_UC_USE_FIXED_PT
void resize_image_uc1_row_proc_implement_sse(unsigned char const*       src_row, 
                                             unsigned char*             dst_row, 
                                             int                        dst_width, 
                                             unsigned short int const*  pos_u, 
                                             unsigned long long         pos_v1,
                                             unsigned long long         pos_v0)
{
    int       j             = 0;
    __m128i   mask32_flag   = _mm_set1_epi32(0x000000FF);
    __m128i   mask32_flag_0 = _mm_set1_epi32(0x00FF0000);
    __m128i   half          = _mm_set1_epi32(8192);
    __m128i   v1            = _mm_set1_epi64x(pos_v1);
    __m128i   v0            = _mm_set1_epi64x(pos_v0);
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0    = _mm_load_si128((__m128i*)src_row);
        __m128i   cur_int8_data00   = _mm_and_si128(int8_data_raw0, mask32_flag);
        __m128i   cur_int8_data02   = _mm_and_si128(_mm_slli_epi32(int8_data_raw0, 8),  mask32_flag_0);
        __m128i   cur_int8_data01   = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_flag);
        __m128i   cur_int8_data03   = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 8),  mask32_flag_0);

        cur_int8_data00             = _mm_or_si128(cur_int8_data00, cur_int8_data02);
        cur_int8_data01             = _mm_or_si128(cur_int8_data01, cur_int8_data03);

        __m128i   u                 = _mm_load_si128((__m128i*)(pos_u));

        __m128i   cur_int8_data000  = _mm_mullo_epi16(u, v1);
        __m128i   cur_int8_data001  = _mm_mulhi_epi16(u, v1);
        __m128i   cur_int8_data002  = _mm_mullo_epi16(u, v0);
        __m128i   cur_int8_data003  = _mm_mulhi_epi16(u, v0);
        __m128i   cur_int8_data004  = _mm_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data005  = _mm_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data006  = _mm_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m128i   cur_int8_data007  = _mm_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data00             = _mm_madd_epi16(cur_int8_data00, cur_int8_data004);
        cur_int8_data01             = _mm_madd_epi16(cur_int8_data01, cur_int8_data006);

        cur_int8_data00             = _mm_add_epi32(cur_int8_data00, cur_int8_data01);
        cur_int8_data00             = _mm_add_epi32(cur_int8_data00, half);
        cur_int8_data00             = _mm_srli_epi32(cur_int8_data00, 14);
        cur_int8_data00             = _mm_packus_epi32(cur_int8_data00, cur_int8_data01);
        cur_int8_data00             = _mm_packus_epi16(cur_int8_data00, cur_int8_data00);

        *((int*)dst_row)            = _mm_cvtsi128_si32(cur_int8_data00);
        src_row                    += 16;
        dst_row                    += 4;
        pos_u                      += 8;
    }
}

void resize_image_uc3_row_proc_implement_sse(unsigned char const*       src_row, 
                                             unsigned char*             dst_row, 
                                             int                        dst_width, 
                                             unsigned short int const*  pos_u, 
                                             unsigned long long         pos_v1,
                                             unsigned long long         pos_v0)
{
    int            j        = 0;
    int            i        = 0;
    const __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    const __m128i  mask32_1 = _mm_set1_epi32(0x00FF0000);
    const __m128i  half     = _mm_set1_epi32(8192);
    const __m128i  UC4TOUC3_SHUFLLE_MASK = _mm_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m128i        v1      = _mm_set1_epi64x(pos_v1);
    __m128i        v0      = _mm_set1_epi64x(pos_v0);
    __m128i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        cur_int8_data0       = _mm_or_si128(_mm_and_si128(int8_data_raw0, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm_or_si128(_mm_and_si128(int8_data_raw2, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw3, mask32_0), 16));

        __m128i   u          = _mm_load_si128((__m128i*)(pos_u));

        __m128i   cur_int8_data000  = _mm_mullo_epi16(u, v1);
        __m128i   cur_int8_data001  = _mm_mulhi_epi16(u, v1);
        __m128i   cur_int8_data002  = _mm_mullo_epi16(u, v0);
        __m128i   cur_int8_data003  = _mm_mulhi_epi16(u, v0);
        __m128i   cur_int8_data004  = _mm_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data005  = _mm_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data006  = _mm_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m128i   cur_int8_data007  = _mm_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res0                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res1                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res2                        = _mm_srli_epi32(cur_int8_data000, 14);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res1                        = _mm_slli_epi32(res1, 8);
        res2                        = _mm_slli_epi32(res2, 16);
        res0                        = _mm_or_si128(res0, res1);
        res0                        = _mm_or_si128(res0, res2);
        
        res1                        = _mm_shuffle_epi8(res0, UC4TOUC3_SHUFLLE_MASK);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)(dst_row))    = _mm_extract_epi64(res1, 0);
#else
        *(int*)(dst_row)            = _mm_extract_epi32(res1,  0);
        *(int*)(dst_row + 4)        = _mm_extract_epi32(res1,  1);
#endif
        *(int*)(dst_row + 8)        = _mm_extract_epi32(res1,  2);
        src_row                    += 64;
        dst_row                    += 12;
        pos_u                      += 8;
    }
}

void resize_image_uc4_row_proc_alpha_fixed_implement_sse(unsigned char const*         src_row, 
                                                         unsigned char*               dst_row, 
                                                         int                          dst_width, 
                                                         unsigned short int const*    pos_u, 
                                                         unsigned long long           pos_v1,
                                                         unsigned long long           pos_v0,
                                                         unsigned char                alpha_value)
{
    int            j        = 0;
    int            i        = 0;
    const __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    const __m128i  mask32_1 = _mm_set1_epi32(0x00FF0000);
    const __m128i  half     = _mm_set1_epi32(8192);
    const __m128i  UC4TOUC3_SHUFLLE_MASK = _mm_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m128i        v1      = _mm_set1_epi64x(pos_v1);
    __m128i        v0      = _mm_set1_epi64x(pos_v0);
    __m128i        alpha   = _mm_set1_epi32((int)(((unsigned int)alpha_value) << 24));
    __m128i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        cur_int8_data0       = _mm_or_si128(_mm_and_si128(int8_data_raw0, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm_or_si128(_mm_and_si128(int8_data_raw2, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw3, mask32_0), 16));

        __m128i   u          = _mm_load_si128((__m128i*)(pos_u));

        __m128i   cur_int8_data000  = _mm_mullo_epi16(u, v1);
        __m128i   cur_int8_data001  = _mm_mulhi_epi16(u, v1);
        __m128i   cur_int8_data002  = _mm_mullo_epi16(u, v0);
        __m128i   cur_int8_data003  = _mm_mulhi_epi16(u, v0);
        __m128i   cur_int8_data004  = _mm_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data005  = _mm_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data006  = _mm_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m128i   cur_int8_data007  = _mm_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res0                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res1                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res2                        = _mm_srli_epi32(cur_int8_data000, 14);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res1                        = _mm_slli_epi32(res1, 8);
        res2                        = _mm_slli_epi32(res2, 16);
        res0                        = _mm_or_si128(res0, res1);
        res0                        = _mm_or_si128(res0, res2);
        
        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res1                        = _mm_slli_epi32(res1,  8);
        res2                        = _mm_slli_epi32(res2, 16);
        res0                        = _mm_or_si128(res0, res1);
        res0                        = _mm_or_si128(res0, res2);
        res0                        = _mm_or_si128(res0, alpha);

        _mm_storeu_si128((__m128i*)dst_row, res0);
        src_row                    += 64;
        dst_row                    += 16;
        pos_u                      += 8;
    }
}

void resize_image_uc4_row_proc_alpha_var_implement_sse(unsigned char const*         src_row, 
                                                       unsigned char*               dst_row, 
                                                       int                          dst_width, 
                                                       unsigned short int const*    pos_u, 
                                                       unsigned long long           pos_v1,
                                                       unsigned long long           pos_v0)
{
    int            j        = 0;
    int            i        = 0;
    const __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    const __m128i  mask32_1 = _mm_set1_epi32(0x00FF0000);
    const __m128i  half     = _mm_set1_epi32(8192);
    const __m128i  UC4TOUC3_SHUFLLE_MASK = _mm_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m128i        v1      = _mm_set1_epi64x(pos_v1);
    __m128i        v0      = _mm_set1_epi64x(pos_v0);
    __m128i  res0, res1, res2, res3;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        cur_int8_data0       = _mm_or_si128(_mm_and_si128(int8_data_raw0, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm_or_si128(_mm_and_si128(int8_data_raw2, mask32_0), 
                                               _mm_slli_epi32(_mm_and_si128(int8_data_raw3, mask32_0), 16));

        __m128i   u          = _mm_load_si128((__m128i*)(pos_u));

        __m128i   cur_int8_data000  = _mm_mullo_epi16(u, v1);
        __m128i   cur_int8_data001  = _mm_mulhi_epi16(u, v1);
        __m128i   cur_int8_data002  = _mm_mullo_epi16(u, v0);
        __m128i   cur_int8_data003  = _mm_mulhi_epi16(u, v0);
        __m128i   cur_int8_data004  = _mm_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data005  = _mm_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m128i   cur_int8_data006  = _mm_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m128i   cur_int8_data007  = _mm_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res0                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                   _mm_and_si128(_mm_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res1                        = _mm_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm_and_si128(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res2                        = _mm_srli_epi32(cur_int8_data000, 14);


        cur_int8_data0              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw0, 24), mask32_0),
                                                   _mm_and_si128(_mm_srli_epi32(int8_data_raw1, 8),  mask32_1));
        cur_int8_data1              = _mm_or_si128(_mm_and_si128(_mm_srli_epi32(int8_data_raw2, 24), mask32_0),
                                                   _mm_and_si128(_mm_srli_epi32(int8_data_raw3, 8),  mask32_1));

        cur_int8_data000            = _mm_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm_add_epi32(cur_int8_data000, half);
        res3                        = _mm_srli_epi32(cur_int8_data000, 14);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res3                        = _mm_min_epi32(res3, mask32_0);
        res1                        = _mm_slli_epi32(res1,  8);
        res2                        = _mm_slli_epi32(res2, 16);
        res3                        = _mm_slli_epi32(res3, 24);
        res0                        = _mm_or_si128(res0, res1);
        res2                        = _mm_or_si128(res2, res3);
        res0                        = _mm_or_si128(res0, res2);
        _mm_storeu_si128((__m128i*)dst_row, res0);
        src_row                    += 64;
        dst_row                    += 16;
        pos_u                      += 8;
    }
}
#else
void resize_image_uc1_row_proc_implement_sse(unsigned char const*  src_row, 
                                             unsigned char*        dst_row, 
                                             int                   dst_width, 
                                             float const*          pos_x, 
                                             float                 pos_y)
{
    int      j        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_y);
    __m128   v1       = _mm_sub_ps(one, v0);
    __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128    u0               = _mm_load_ps(pos_x);
        __m128i   cur_int8_data00  = _mm_and_si128(int8_data_raw0, mask32_0);
        __m128i   cur_int8_data01  = _mm_and_si128(_mm_srli_epi32(int8_data_raw0,  8), mask32_0);
        __m128i   cur_int8_data10  = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0);
        __m128i   cur_int8_data11  = _mm_srli_epi32(int8_data_raw0, 24);
        __m128    data0            = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1            = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data2            = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data3            = _mm_cvtepi32_ps(cur_int8_data11);
        __m128    u1               = _mm_sub_ps(one, u0);

        __m128    res1             = _mm_mul_ps(u0, v1);
        __m128    res3             = _mm_mul_ps(u0, v0);
        __m128    res0             = _mm_mul_ps(u1, v1);
        __m128    res2             = _mm_mul_ps(u1, v0);

        res0                       = _mm_mul_ps(data0, res0);
        res1                       = _mm_mul_ps(data1, res1);
        res2                       = _mm_mul_ps(data2, res2);
        res3                       = _mm_mul_ps(data3, res3);
        res0                       = _mm_add_ps(res0, res1);
        res2                       = _mm_add_ps(res2, res3);
        res0                       = _mm_add_ps(res0, res2);
        int8_data_raw0             = _mm_cvtps_epi32(res0);

        int8_data_raw0             = _mm_packus_epi32(int8_data_raw0, int8_data_raw0);
        int8_data_raw0             = _mm_packus_epi16(int8_data_raw0, int8_data_raw0);
        *((unsigned int*)dst_row)  = _mm_cvtsi128_si32(int8_data_raw0);
        src_row                   += 16;
        dst_row                   += 4;
        pos_x                     += 4;
    }
}

void resize_image_uc3_row_proc_implement_sse(unsigned char const*  src_row, 
                                             unsigned char*        dst_row, 
                                             int                   dst_width, 
                                             float const*          pos_x, 
                                             float                 pos_y)
{
    int      j        = 0;
    int      i        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_y);
    __m128   v1       = _mm_sub_ps(one, v0);
    const __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    const __m128i  UC4TOUC3_SHUFLLE_MASK = _mm_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m128i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        __m128    u0                = _mm_load_ps(pos_x);
        __m128i   cur_int8_data00   = _mm_and_si128(int8_data_raw0, mask32_0);
        __m128i   cur_int8_data01   = _mm_and_si128(int8_data_raw1, mask32_0);
        __m128i   cur_int8_data10   = _mm_and_si128(int8_data_raw2, mask32_0);
        __m128i   cur_int8_data11   = _mm_and_si128(int8_data_raw3, mask32_0);

        __m128    data0             = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1             = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data2             = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data3             = _mm_cvtepi32_ps(cur_int8_data11);
        __m128    u1                = _mm_sub_ps(one, u0);

        __m128    coef1             = _mm_mul_ps(u0, v1);
        __m128    coef3             = _mm_mul_ps(u0, v0);
        __m128    coef0             = _mm_mul_ps(u1, v1);
        __m128    coef2             = _mm_mul_ps(u1, v0);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res0                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res1                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res2                        = _mm_cvtps_epi32(data0);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res1                        = _mm_slli_epi32(res1,  8);
        res2                        = _mm_slli_epi32(res2, 16);
        res0                        = _mm_or_si128(res0, res1);
        res0                        = _mm_or_si128(res0, res2);
        
        res1                        = _mm_shuffle_epi8(res0, UC4TOUC3_SHUFLLE_MASK);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)(dst_row))    = _mm_extract_epi64(res1, 0);
#else
        *(int*)(dst_row)            = _mm_extract_epi32(res1,  0);
        *(int*)(dst_row + 4)        = _mm_extract_epi32(res1,  1);
#endif
        *(int*)(dst_row + 8)        = _mm_extract_epi32(res1,  2);
        src_row                    += 64;
        dst_row                    += 12;
        pos_x                      += 4;
    }
}

void resize_image_uc4_row_proc_alpha_fixed_implement_sse(unsigned char const*  src_row, 
                                                         unsigned char*        dst_row, 
                                                         int                   dst_width, 
                                                         float const*          pos_x, 
                                                         float                 pos_y,
                                                         unsigned char         alpha_value)
{
    int      j        = 0;
    int      i        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_y);
    __m128   v1       = _mm_sub_ps(one, v0);
    __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    __m128i  alpha    = _mm_set1_epi32((int)(((unsigned int)alpha_value) << 24));
    __m128i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        __m128    u0                = _mm_load_ps(pos_x);
        __m128i   cur_int8_data00   = _mm_and_si128(int8_data_raw0, mask32_0);
        __m128i   cur_int8_data01   = _mm_and_si128(int8_data_raw1, mask32_0);
        __m128i   cur_int8_data10   = _mm_and_si128(int8_data_raw2, mask32_0);
        __m128i   cur_int8_data11   = _mm_and_si128(int8_data_raw3, mask32_0);

        __m128    data0             = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1             = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data2             = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data3             = _mm_cvtepi32_ps(cur_int8_data11);
        __m128    u1                = _mm_sub_ps(one, u0);

        __m128    coef1             = _mm_mul_ps(u0, v1);
        __m128    coef3             = _mm_mul_ps(u0, v0);
        __m128    coef0             = _mm_mul_ps(u1, v1);
        __m128    coef2             = _mm_mul_ps(u1, v0);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res0                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res1                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res2                        = _mm_cvtps_epi32(data0);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res1                        = _mm_slli_epi32(res1,  8);
        res2                        = _mm_slli_epi32(res2, 16);
        res0                        = _mm_or_si128(res0, res1);
        res0                        = _mm_or_si128(res0, res2);
        res0                        = _mm_or_si128(res0, alpha);

        _mm_storeu_si128((__m128i*)dst_row, res0);
        src_row                    += 64;
        dst_row                    += 16;
        pos_x                      += 4;
    }
}

void resize_image_uc4_row_proc_alpha_var_implement_sse(unsigned char const*  src_row, 
                                                       unsigned char*        dst_row, 
                                                       int                   dst_width, 
                                                       float const*          pos_x, 
                                                       float                 pos_y)
{
    int      j        = 0;
    int      i        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_y);
    __m128   v1       = _mm_sub_ps(one, v0);
    __m128i  mask32_0 = _mm_set1_epi32(0x000000FF);
    __m128i  res0, res1, res2, res3;
    for (j = 0; j < dst_width; j += 4)
    {
        __m128i   int8_data_raw0   = _mm_load_si128((__m128i*)src_row);
        __m128i   int8_data_raw1   = _mm_load_si128((__m128i*)(src_row + 16));
        __m128i   int8_data_raw2   = _mm_load_si128((__m128i*)(src_row + 32));
        __m128i   int8_data_raw3   = _mm_load_si128((__m128i*)(src_row + 48));
        __m128i   cur_int8_data0, cur_int8_data1, cur_int8_data2, cur_int8_data3;
        {
            cur_int8_data0   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data2   = _mm_castps_si128(_mm_unpacklo_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));
            cur_int8_data1   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw0), _mm_castsi128_ps(int8_data_raw1)));
            cur_int8_data3   = _mm_castps_si128(_mm_unpackhi_ps(_mm_castsi128_ps(int8_data_raw2), _mm_castsi128_ps(int8_data_raw3)));

            int8_data_raw0   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data0), _mm_castsi128_ps(cur_int8_data2)));
            int8_data_raw1   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data2), _mm_castsi128_ps(cur_int8_data0)));
            int8_data_raw2   = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(cur_int8_data1), _mm_castsi128_ps(cur_int8_data3)));
            int8_data_raw3   = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(cur_int8_data3), _mm_castsi128_ps(cur_int8_data1)));
        }

        __m128    u0                = _mm_load_ps(pos_x);
        __m128i   cur_int8_data00   = _mm_and_si128(int8_data_raw0, mask32_0);
        __m128i   cur_int8_data01   = _mm_and_si128(int8_data_raw1, mask32_0);
        __m128i   cur_int8_data10   = _mm_and_si128(int8_data_raw2, mask32_0);
        __m128i   cur_int8_data11   = _mm_and_si128(int8_data_raw3, mask32_0);

        __m128    data0             = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1             = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data2             = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data3             = _mm_cvtepi32_ps(cur_int8_data11);
        __m128    u1                = _mm_sub_ps(one, u0);

        __m128    coef1             = _mm_mul_ps(u0, v1);
        __m128    coef3             = _mm_mul_ps(u0, v0);
        __m128    coef0             = _mm_mul_ps(u1, v1);
        __m128    coef2             = _mm_mul_ps(u1, v0);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res0                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res1                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res2                        = _mm_cvtps_epi32(data0);

        cur_int8_data00             = _mm_and_si128(_mm_srli_epi32(int8_data_raw0, 24), mask32_0);
        cur_int8_data01             = _mm_and_si128(_mm_srli_epi32(int8_data_raw1, 24), mask32_0);
        cur_int8_data10             = _mm_and_si128(_mm_srli_epi32(int8_data_raw2, 24), mask32_0);
        cur_int8_data11             = _mm_and_si128(_mm_srli_epi32(int8_data_raw3, 24), mask32_0);
        data0                       = _mm_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm_mul_ps(data0, coef0);
        data1                       = _mm_mul_ps(data1, coef1);
        data2                       = _mm_mul_ps(data2, coef2);
        data3                       = _mm_mul_ps(data3, coef3);
        data0                       = _mm_add_ps(data0, data1);
        data2                       = _mm_add_ps(data2, data3);
        data0                       = _mm_add_ps(data0, data2);
        res3                        = _mm_cvtps_epi32(data0);

        res0                        = _mm_min_epi32(res0, mask32_0);
        res1                        = _mm_min_epi32(res1, mask32_0);
        res2                        = _mm_min_epi32(res2, mask32_0);
        res3                        = _mm_min_epi32(res3, mask32_0);
        res1                        = _mm_slli_epi32(res1,  8);
        res2                        = _mm_slli_epi32(res2, 16);
        res3                        = _mm_slli_epi32(res3, 24);
        res0                        = _mm_or_si128(res0, res1);
        res2                        = _mm_or_si128(res2, res3);
        res0                        = _mm_or_si128(res0, res2);

        _mm_storeu_si128((__m128i*)dst_row, res0);
        src_row                    += 64;
        dst_row                    += 16;
        pos_x                      += 4;
    }
}
#endif

void resize_image_f1_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_v);
    __m128   v1       = _mm_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 4)
    {
        __m128   data_raw0   = _mm_load_ps(src_row);
        __m128   data_raw1   = _mm_load_ps((src_row + 4));
        __m128   data_raw2   = _mm_load_ps((src_row + 8));
        __m128   data_raw3   = _mm_load_ps((src_row + 12));
        __m128   u0          = _mm_load_ps(pos_u);
        __m128   cur_data0, cur_data2, cur_data1, cur_data3;
        {
            cur_data0   = _mm_unpacklo_ps(data_raw0, data_raw1);
            cur_data2   = _mm_unpacklo_ps(data_raw2, data_raw3);
            cur_data1   = _mm_unpackhi_ps(data_raw0, data_raw1);
            cur_data3   = _mm_unpackhi_ps(data_raw2, data_raw3);

            data_raw0   = _mm_movelh_ps(cur_data0, cur_data2);
            data_raw1   = _mm_movehl_ps(cur_data2, cur_data0);
            data_raw2   = _mm_movelh_ps(cur_data1, cur_data3);
            data_raw3   = _mm_movehl_ps(cur_data3, cur_data1);
        }

        __m128    u1               = _mm_sub_ps(one, u0);
        __m128    res1             = _mm_mul_ps(u0, v1);
        __m128    res3             = _mm_mul_ps(u0, v0);
        __m128    res0             = _mm_mul_ps(u1, v1);
        __m128    res2             = _mm_mul_ps(u1, v0);

        res0                       = _mm_mul_ps(data_raw0, res0);
        res1                       = _mm_mul_ps(data_raw1, res1);
        res2                       = _mm_mul_ps(data_raw2, res2);
        res3                       = _mm_mul_ps(data_raw3, res3);
        res0                       = _mm_add_ps(res0, res1);
        res2                       = _mm_add_ps(res2, res3);
        res0                       = _mm_add_ps(res0, res2);
        
        _mm_storeu_ps(dst_row, res0);   
        src_row                   += 16;
        dst_row                   += 4;
        pos_u                     += 4;
    }
}

void resize_image_f3_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_v);
    __m128   v1       = _mm_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 2)
    {
        __m128   data_raw0   = _mm_load_ps(src_row);
        __m128   data_raw1   = _mm_load_ps((src_row + 4));
        __m128   data_raw2   = _mm_load_ps((src_row + 8));
        __m128   data_raw3   = _mm_load_ps((src_row + 12));
        __m128   data_raw4   = _mm_load_ps((src_row + 16));
        __m128   data_raw5   = _mm_load_ps((src_row + 20));
        __m128   data_raw6   = _mm_load_ps((src_row + 24));
        __m128   data_raw7   = _mm_load_ps((src_row + 28));
        __m128   u00         = _mm_set1_ps(pos_u[0]);
        __m128   u01         = _mm_set1_ps(pos_u[1]);

        __m128   u10         = _mm_sub_ps(one, u00);
        __m128   u11         = _mm_sub_ps(one, u01);

        __m128    coef00     = _mm_mul_ps(u10, v1);
        __m128    coef01     = _mm_mul_ps(u00, v1);
        __m128    coef02     = _mm_mul_ps(u10, v0);
        __m128    coef03     = _mm_mul_ps(u00, v0);

        __m128    coef04     = _mm_mul_ps(u11, v1);
        __m128    coef05     = _mm_mul_ps(u01, v1);
        __m128    coef06     = _mm_mul_ps(u11, v0);
        __m128    coef07     = _mm_mul_ps(u01, v0);

        coef00               = _mm_mul_ps(data_raw0, coef00);
        coef01               = _mm_mul_ps(data_raw1, coef01);
        coef02               = _mm_mul_ps(data_raw2, coef02);
        coef03               = _mm_mul_ps(data_raw3, coef03);

        coef04               = _mm_mul_ps(data_raw4, coef04);
        coef05               = _mm_mul_ps(data_raw5, coef05);
        coef06               = _mm_mul_ps(data_raw6, coef06);
        coef07               = _mm_mul_ps(data_raw7, coef07);

        coef00               = _mm_add_ps(coef00, coef01);
        coef02               = _mm_add_ps(coef02, coef03);
        coef04               = _mm_add_ps(coef04, coef05);
        coef06               = _mm_add_ps(coef06, coef07);
        coef00               = _mm_add_ps(coef00, coef02);
        coef04               = _mm_add_ps(coef04, coef06);
        
        _mm_storeu_ps(dst_row,     coef00);
#if defined(__x86_64__) || defined(_M_X64)
        *(long long*)(dst_row + 3) = _mm_extract_epi64(_mm_castps_si128(coef04), 0);
#else
        *(int*)(dst_row + 3)       = _mm_extract_epi32(_mm_castps_si128(coef04), 0);
        *(int*)(dst_row + 4)       = _mm_extract_epi32(_mm_castps_si128(coef04), 1);
#endif
        *(int*)(dst_row + 5)       = _mm_extract_epi32(_mm_castps_si128(coef04), 2);

        src_row                   += 32;
        dst_row                   += 6;
        pos_u                     += 2;
    }
}

void resize_image_f4_row_proc_implement_sse(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m128   one      = _mm_set1_ps(1.0f);
    __m128   v0       = _mm_set1_ps(pos_v);
    __m128   v1       = _mm_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 2)
    {
        __m128   data_raw0   = _mm_load_ps(src_row);
        __m128   data_raw1   = _mm_load_ps((src_row + 4));
        __m128   data_raw2   = _mm_load_ps((src_row + 8));
        __m128   data_raw3   = _mm_load_ps((src_row + 12));
        __m128   data_raw4   = _mm_load_ps((src_row + 16));
        __m128   data_raw5   = _mm_load_ps((src_row + 20));
        __m128   data_raw6   = _mm_load_ps((src_row + 24));
        __m128   data_raw7   = _mm_load_ps((src_row + 28));
        __m128   u00         = _mm_set1_ps(pos_u[0]);
        __m128   u01         = _mm_set1_ps(pos_u[1]);

        __m128   u10         = _mm_sub_ps(one, u00);
        __m128   u11         = _mm_sub_ps(one, u01);

        __m128    coef00     = _mm_mul_ps(u10, v1);
        __m128    coef01     = _mm_mul_ps(u00, v1);
        __m128    coef02     = _mm_mul_ps(u10, v0);
        __m128    coef03     = _mm_mul_ps(u00, v0);

        __m128    coef04     = _mm_mul_ps(u11, v1);
        __m128    coef05     = _mm_mul_ps(u01, v1);
        __m128    coef06     = _mm_mul_ps(u11, v0);
        __m128    coef07     = _mm_mul_ps(u01, v0);

        coef00               = _mm_mul_ps(data_raw0, coef00);
        coef01               = _mm_mul_ps(data_raw1, coef01);
        coef02               = _mm_mul_ps(data_raw2, coef02);
        coef03               = _mm_mul_ps(data_raw3, coef03);

        coef04               = _mm_mul_ps(data_raw4, coef04);
        coef05               = _mm_mul_ps(data_raw5, coef05);
        coef06               = _mm_mul_ps(data_raw6, coef06);
        coef07               = _mm_mul_ps(data_raw7, coef07);

        coef00               = _mm_add_ps(coef00, coef01);
        coef02               = _mm_add_ps(coef02, coef03);
        coef04               = _mm_add_ps(coef04, coef05);
        coef06               = _mm_add_ps(coef06, coef07);
        coef00               = _mm_add_ps(coef00, coef02);
        coef04               = _mm_add_ps(coef04, coef06);
        
        _mm_storeu_ps(dst_row,     coef00);
        _mm_storeu_ps(dst_row + 4, coef04);
        src_row                   += 32;
        dst_row                   += 8;
        pos_u                     += 2;
    }
}
