#include <immintrin.h>
#include "../../../../common/dlcv_proc_opt_com_def.h"

#ifdef RESIZE_UC_USE_FIXED_PT
void resize_image_uc1_row_proc_implement_avx(unsigned char const*       src_row, 
                                             unsigned char*             dst_row, 
                                             int                        dst_width, 
                                             unsigned short int const*  pos_u, 
                                             unsigned long long         pos_v1,
                                             unsigned long long         pos_v0)
{
    int             j             = 0;
    const __m256i   mask32_flag   = _mm256_set1_epi32(0x000000FF);
    const __m256i   mask32_flag_0 = _mm256_set1_epi32(0x00FF0000);
    const __m256i   half          = _mm256_set1_epi32(8192);
    __m256i         v1            = _mm256_set1_epi64x(pos_v1);
    __m256i         v0            = _mm256_set1_epi64x(pos_v0);
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0    = _mm256_load_si256((__m256i*)src_row);
        __m256i   cur_int8_data00   = _mm256_and_si256(int8_data_raw0, mask32_flag);
        __m256i   cur_int8_data02   = _mm256_and_si256(_mm256_slli_epi32(int8_data_raw0, 8),  mask32_flag_0);
        __m256i   cur_int8_data01   = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_flag);
        __m256i   cur_int8_data03   = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 8),  mask32_flag_0);

        cur_int8_data00             = _mm256_or_si256(cur_int8_data00, cur_int8_data02);
        cur_int8_data01             = _mm256_or_si256(cur_int8_data01, cur_int8_data03);

        __m256i   u                 = _mm256_load_si256((__m256i*)(pos_u));

        __m256i   cur_int8_data000  = _mm256_mullo_epi16(u, v1);
        __m256i   cur_int8_data001  = _mm256_mulhi_epi16(u, v1);
        __m256i   cur_int8_data002  = _mm256_mullo_epi16(u, v0);
        __m256i   cur_int8_data003  = _mm256_mulhi_epi16(u, v0);
        __m256i   cur_int8_data004  = _mm256_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data005  = _mm256_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data006  = _mm256_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m256i   cur_int8_data007  = _mm256_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm256_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm256_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm256_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm256_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm256_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm256_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data00             = _mm256_madd_epi16(cur_int8_data00, cur_int8_data004);
        cur_int8_data01             = _mm256_madd_epi16(cur_int8_data01, cur_int8_data006);

        cur_int8_data00             = _mm256_add_epi32(cur_int8_data00, cur_int8_data01);
        cur_int8_data00             = _mm256_add_epi32(cur_int8_data00, half);
        cur_int8_data00             = _mm256_srli_epi32(cur_int8_data00, 14);
        cur_int8_data01             = _mm256_permute2f128_si256(cur_int8_data00, cur_int8_data00, 1);
        cur_int8_data00             = _mm256_packus_epi32(cur_int8_data00, cur_int8_data01);
        cur_int8_data00             = _mm256_packus_epi16(cur_int8_data00, cur_int8_data00);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)dst_row)      = _mm256_extract_epi64(cur_int8_data00, 0);
#else
        *((int*)(dst_row))          = _mm256_extract_epi32(cur_int8_data00, 0);
        *((int*)(dst_row + 4))      = _mm256_extract_epi32(cur_int8_data00, 1);
#endif
        src_row                    += 32;
        dst_row                    += 8;
        pos_u                      += 16;
    }
}
void resize_image_uc3_row_proc_implement_avx(unsigned char const*       src_row, 
                                             unsigned char*             dst_row, 
                                             int                        dst_width, 
                                             unsigned short int const*  pos_u, 
                                             unsigned long long         pos_v1,
                                             unsigned long long         pos_v0)
{
    int            j        = 0;
    int            i        = 0;
    const __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    const __m256i  mask32_1 = _mm256_set1_epi32(0x00FF0000);
    const __m256i  half     = _mm256_set1_epi32(8192);
    const __m256i  UC4TOUC3_SHUFLLE_MASK = _mm256_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100,
                                                            0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m256i         v1      = _mm256_set1_epi64x(pos_v1);
    __m256i         v0      = _mm256_set1_epi64x(pos_v0);
    __m256i  res0, res1, res2;
    __m128i  res4;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }
        cur_int8_data0       = _mm256_or_si256(_mm256_and_si256(int8_data_raw0, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm256_or_si256(_mm256_and_si256(int8_data_raw2, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw3, mask32_0), 16));

        __m256i   u                 = _mm256_load_si256((__m256i*)(pos_u));

        __m256i   cur_int8_data000  = _mm256_mullo_epi16(u, v1);
        __m256i   cur_int8_data001  = _mm256_mulhi_epi16(u, v1);
        __m256i   cur_int8_data002  = _mm256_mullo_epi16(u, v0);
        __m256i   cur_int8_data003  = _mm256_mulhi_epi16(u, v0);
        __m256i   cur_int8_data004  = _mm256_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data005  = _mm256_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data006  = _mm256_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m256i   cur_int8_data007  = _mm256_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm256_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm256_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm256_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm256_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm256_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm256_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res0                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res1                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res2                        = _mm256_srli_epi32(cur_int8_data000, 14);

        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res1                        = _mm256_slli_epi32(res1, 8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res0                        = _mm256_or_si256(res0, res1);
        res0                        = _mm256_or_si256(res0, res2);
        
        res1                          = _mm256_shuffle_epi8(res0, UC4TOUC3_SHUFLLE_MASK);
        res4                          = _mm256_castsi256_si128(res1);
        _mm_storeu_si128((__m128i*)(dst_row), res4);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)(dst_row + 12)) = _mm256_extract_epi64(res1, 2);
#else
        *(int*)(dst_row + 12)         = _mm256_extract_epi32(res1,  4);
        *(int*)(dst_row + 16)         = _mm256_extract_epi32(res1,  5);
#endif
        *(int*)(dst_row + 20)         = _mm256_extract_epi32(res1,  6);
        src_row                      += 128;
        dst_row                      += 24;
        pos_u                        += 16;
    }
}

void resize_image_uc4_row_proc_alpha_fixed_implement_avx(unsigned char const*         src_row, 
                                                         unsigned char*               dst_row, 
                                                         int                          dst_width, 
                                                         unsigned short int const*    pos_u, 
                                                         unsigned long long           pos_v1,
                                                         unsigned long long           pos_v0,
                                                         unsigned char                alpha_value)
{
    int            j        = 0;
    int            i        = 0;
    const __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    const __m256i  mask32_1 = _mm256_set1_epi32(0x00FF0000);
    const __m256i  half     = _mm256_set1_epi32(8192);
    const __m256i  UC4TOUC3_SHUFLLE_MASK = _mm256_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100,
                                                            0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m256i        v1       = _mm256_set1_epi64x(pos_v1);
    __m256i        v0       = _mm256_set1_epi64x(pos_v0);
    __m256i        alpha    = _mm256_set1_epi32((int)(((unsigned int)alpha_value) << 24));
    __m256i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }
        cur_int8_data0       = _mm256_or_si256(_mm256_and_si256(int8_data_raw0, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm256_or_si256(_mm256_and_si256(int8_data_raw2, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw3, mask32_0), 16));

        __m256i   u                 = _mm256_load_si256((__m256i*)(pos_u));

        __m256i   cur_int8_data000  = _mm256_mullo_epi16(u, v1);
        __m256i   cur_int8_data001  = _mm256_mulhi_epi16(u, v1);
        __m256i   cur_int8_data002  = _mm256_mullo_epi16(u, v0);
        __m256i   cur_int8_data003  = _mm256_mulhi_epi16(u, v0);
        __m256i   cur_int8_data004  = _mm256_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data005  = _mm256_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data006  = _mm256_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m256i   cur_int8_data007  = _mm256_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm256_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm256_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm256_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm256_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm256_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm256_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res0                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res1                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res2                        = _mm256_srli_epi32(cur_int8_data000, 14);

        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res1                        = _mm256_slli_epi32(res1, 8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res0                        = _mm256_or_si256(res0, res1);
        res2                        = _mm256_or_si256(res2, alpha);
        res0                        = _mm256_or_si256(res0, res2);
        
        _mm256_storeu_si256((__m256i*)dst_row, res0);
        src_row                    += 128;
        dst_row                    += 32;
        pos_u                      += 16;
    }
}

void resize_image_uc4_row_proc_alpha_var_implement_avx(unsigned char const*         src_row, 
                                                       unsigned char*               dst_row, 
                                                       int                          dst_width, 
                                                       unsigned short int const*    pos_u, 
                                                       unsigned long long           pos_v1,
                                                       unsigned long long           pos_v0)
{
    int            j        = 0;
    int            i        = 0;
    const __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    const __m256i  mask32_1 = _mm256_set1_epi32(0x00FF0000);
    const __m256i  half     = _mm256_set1_epi32(8192);
    const __m256i  UC4TOUC3_SHUFLLE_MASK = _mm256_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100,
                                                            0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m256i        v1       = _mm256_set1_epi64x(pos_v1);
    __m256i        v0       = _mm256_set1_epi64x(pos_v0);
    __m256i  res0, res1, res2, res3;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }
        cur_int8_data0       = _mm256_or_si256(_mm256_and_si256(int8_data_raw0, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw1, mask32_0), 16));
        cur_int8_data1       = _mm256_or_si256(_mm256_and_si256(int8_data_raw2, mask32_0), 
                                               _mm256_slli_epi32(_mm256_and_si256(int8_data_raw3, mask32_0), 16));

        __m256i   u                 = _mm256_load_si256((__m256i*)(pos_u));

        __m256i   cur_int8_data000  = _mm256_mullo_epi16(u, v1);
        __m256i   cur_int8_data001  = _mm256_mulhi_epi16(u, v1);
        __m256i   cur_int8_data002  = _mm256_mullo_epi16(u, v0);
        __m256i   cur_int8_data003  = _mm256_mulhi_epi16(u, v0);
        __m256i   cur_int8_data004  = _mm256_unpacklo_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data005  = _mm256_unpackhi_epi16(cur_int8_data000, cur_int8_data001); 
        __m256i   cur_int8_data006  = _mm256_unpacklo_epi16(cur_int8_data002, cur_int8_data003); 
        __m256i   cur_int8_data007  = _mm256_unpackhi_epi16(cur_int8_data002, cur_int8_data003); 

        cur_int8_data004            = _mm256_srli_epi32(cur_int8_data004, 8);
        cur_int8_data005            = _mm256_srli_epi32(cur_int8_data005, 8);
        cur_int8_data006            = _mm256_srli_epi32(cur_int8_data006, 8);
        cur_int8_data007            = _mm256_srli_epi32(cur_int8_data007, 8);

        cur_int8_data004            = _mm256_packus_epi32(cur_int8_data004, cur_int8_data005);
        cur_int8_data006            = _mm256_packus_epi32(cur_int8_data006, cur_int8_data007);

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res0                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw1, 8), mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 8), mask32_0),
                                                      _mm256_and_si256(_mm256_slli_epi32(int8_data_raw3, 8), mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res1                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw1, mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0),
                                                      _mm256_and_si256(int8_data_raw3, mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res2                        = _mm256_srli_epi32(cur_int8_data000, 14);

        cur_int8_data0              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 24), mask32_0),
                                                      _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1, 8),  mask32_1));
        cur_int8_data1              = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 24), mask32_0),
                                                      _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3, 8),  mask32_1));

        cur_int8_data000            = _mm256_madd_epi16(cur_int8_data0, cur_int8_data004);
        cur_int8_data001            = _mm256_madd_epi16(cur_int8_data1, cur_int8_data006);

        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, cur_int8_data001);
        cur_int8_data000            = _mm256_add_epi32(cur_int8_data000, half);
        res3                        = _mm256_srli_epi32(cur_int8_data000, 14);


        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res1                        = _mm256_slli_epi32(res1,  8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res3                        = _mm256_slli_epi32(res3, 24);
        res0                        = _mm256_or_si256(res0, res1);
        res2                        = _mm256_or_si256(res2, res3);
        res0                        = _mm256_or_si256(res0, res2);
        
        _mm256_storeu_si256((__m256i*)dst_row, res0);
        src_row                    += 128;
        dst_row                    += 32;
        pos_u                      += 16;
    }
}

#else
void resize_image_uc1_row_proc_implement_avx(unsigned char const*  src_row, 
                                             unsigned char*        dst_row, 
                                             int                   dst_width, 
                                             float const*          pos_x, 
                                             float                 pos_y)
{
    int      j        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_y);
    __m256   v1       = _mm256_sub_ps(one, v0);
    __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256    u0               = _mm256_load_ps(pos_x);
        __m256i   cur_int8_data00  = _mm256_and_si256(int8_data_raw0, mask32_0);
        __m256i   cur_int8_data01  = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0,  8), mask32_0);
        __m256i   cur_int8_data10  = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0);
        __m256i   cur_int8_data11  = _mm256_srli_epi32(int8_data_raw0, 24);
        __m256    data0            = _mm256_cvtepi32_ps(cur_int8_data00);
        __m256    data1            = _mm256_cvtepi32_ps(cur_int8_data01);
        __m256    data2            = _mm256_cvtepi32_ps(cur_int8_data10);
        __m256    data3            = _mm256_cvtepi32_ps(cur_int8_data11);
        __m256    u1               = _mm256_sub_ps(one, u0);

        __m256    res1             = _mm256_mul_ps(u0, v1);
        __m256    res3             = _mm256_mul_ps(u0, v0);
        __m256    res0             = _mm256_mul_ps(u1, v1);
        __m256    res2             = _mm256_mul_ps(u1, v0);

        res0                       = _mm256_mul_ps(data0, res0);
        res1                       = _mm256_mul_ps(data1, res1);
        res2                       = _mm256_mul_ps(data2, res2);
        res3                       = _mm256_mul_ps(data3, res3);
        res0                       = _mm256_add_ps(res0, res1);
        res2                       = _mm256_add_ps(res2, res3);
        res0                       = _mm256_add_ps(res0, res2);
        int8_data_raw0             = _mm256_cvtps_epi32(res0);
        
        cur_int8_data00            = _mm256_permute2f128_si256(int8_data_raw0, int8_data_raw0, 1);
        int8_data_raw0             = _mm256_packus_epi32(int8_data_raw0, cur_int8_data00);
        int8_data_raw0             = _mm256_packus_epi16(int8_data_raw0, int8_data_raw0);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)dst_row)     = _mm256_extract_epi64(int8_data_raw0, 0);
#else
        *((int*)dst_row)           = _mm256_extract_epi32(int8_data_raw0, 0);
        *((int*)(dst_row + 4))     = _mm256_extract_epi32(int8_data_raw0, 1);
#endif        
        src_row                   += 32;
        dst_row                   += 8;
        pos_x                     += 8;
    }
}

void resize_image_uc3_row_proc_implement_avx(unsigned char const*  src_row, 
                                             unsigned char*        dst_row, 
                                             int                   dst_width, 
                                             float const*          pos_x, 
                                             float                 pos_y)
{
    int            j        = 0;
    int            i        = 0;
    __m256         one      = _mm256_set1_ps(1.0f);
    __m256         v0       = _mm256_set1_ps(pos_y);
    __m256         v1       = _mm256_sub_ps(one, v0);
    const __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    const __m256i  UC4TOUC3_SHUFLLE_MASK = _mm256_set_epi32(0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100,
                                                            0xFFFFFFFF, 0x0E0D0C0A, 0x09080605, 0x04020100);
    __m256i  res0, res1, res2;
    __m128i  res4;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }

        __m256    u0                = _mm256_load_ps(pos_x);
        __m256i   cur_int8_data00   = _mm256_and_si256(int8_data_raw0, mask32_0);
        __m256i   cur_int8_data01   = _mm256_and_si256(int8_data_raw1, mask32_0);
        __m256i   cur_int8_data10   = _mm256_and_si256(int8_data_raw2, mask32_0);
        __m256i   cur_int8_data11   = _mm256_and_si256(int8_data_raw3, mask32_0);

        __m256    data0             = _mm256_cvtepi32_ps(cur_int8_data00);
        __m256    data1             = _mm256_cvtepi32_ps(cur_int8_data01);
        __m256    data2             = _mm256_cvtepi32_ps(cur_int8_data10);
        __m256    data3             = _mm256_cvtepi32_ps(cur_int8_data11);
        __m256    u1                = _mm256_sub_ps(one, u0);

        __m256    coef1             = _mm256_mul_ps(u0, v1);
        __m256    coef3             = _mm256_mul_ps(u0, v0);
        __m256    coef0             = _mm256_mul_ps(u1, v1);
        __m256    coef2             = _mm256_mul_ps(u1, v0);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        //data0                       = _mm256_round_ps(data0, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
        res0                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        //data0                       = _mm256_round_ps(data0, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
        res1                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        //data0                       = _mm256_round_ps(data0, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
        res2                        = _mm256_cvtps_epi32(data0);

        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res1                        = _mm256_slli_epi32(res1, 8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res0                        = _mm256_or_si256(res0, res1);
        res0                        = _mm256_or_si256(res0, res2);
        
        res1                          = _mm256_shuffle_epi8(res0, UC4TOUC3_SHUFLLE_MASK);
        res4                          = _mm256_castsi256_si128(res1);
        _mm_storeu_si128((__m128i*)(dst_row), res4);
#if defined(__x86_64__) || defined(_M_X64)
        *((long long*)(dst_row + 12)) = _mm256_extract_epi64(res1, 2);
#else
        *(int*)(dst_row + 12)         = _mm256_extract_epi32(res1,  4);
        *(int*)(dst_row + 16)         = _mm256_extract_epi32(res1,  5);
#endif
        *(int*)(dst_row + 20)         = _mm256_extract_epi32(res1,  6);
        src_row                    += 128;
        dst_row                    += 24;
        pos_x                      += 8;
    }
}

void resize_image_uc4_row_proc_alpha_fixed_implement_avx(unsigned char const*  src_row, 
                                                         unsigned char*        dst_row, 
                                                         int                   dst_width, 
                                                         float const*          pos_x, 
                                                         float                 pos_y,
                                                         unsigned char         alpha_value)
{
    int      j        = 0;
    int      i        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_y);
    __m256   v1       = _mm256_sub_ps(one, v0);
    __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    __m256i  alpha    = _mm256_set1_epi32((int)(((unsigned int)alpha_value) << 24));
    __m256i  res0, res1, res2;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }

        __m256    u0                = _mm256_load_ps(pos_x);
        __m256i   cur_int8_data00   = _mm256_and_si256(int8_data_raw0, mask32_0);
        __m256i   cur_int8_data01   = _mm256_and_si256(int8_data_raw1, mask32_0);
        __m256i   cur_int8_data10   = _mm256_and_si256(int8_data_raw2, mask32_0);
        __m256i   cur_int8_data11   = _mm256_and_si256(int8_data_raw3, mask32_0);

        __m256    data0             = _mm256_cvtepi32_ps(cur_int8_data00);
        __m256    data1             = _mm256_cvtepi32_ps(cur_int8_data01);
        __m256    data2             = _mm256_cvtepi32_ps(cur_int8_data10);
        __m256    data3             = _mm256_cvtepi32_ps(cur_int8_data11);
        __m256    u1                = _mm256_sub_ps(one, u0);

        __m256    coef1             = _mm256_mul_ps(u0, v1);
        __m256    coef3             = _mm256_mul_ps(u0, v0);
        __m256    coef0             = _mm256_mul_ps(u1, v1);
        __m256    coef2             = _mm256_mul_ps(u1, v0);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res0                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res1                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res2                        = _mm256_cvtps_epi32(data0);

        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res1                        = _mm256_slli_epi32(res1, 8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res0                        = _mm256_or_si256(res0, res1);
        res0                        = _mm256_or_si256(res0, res2);
        res0                        = _mm256_or_si256(res0, alpha);

        _mm256_storeu_si256((__m256i*)dst_row, res0);
        src_row                    += 128;
        dst_row                    += 32;
        pos_x                      += 8;
    }
}

void resize_image_uc4_row_proc_alpha_var_implement_avx(unsigned char const*  src_row, 
                                                       unsigned char*        dst_row, 
                                                       int                   dst_width, 
                                                       float const*          pos_x, 
                                                       float                 pos_y)
{
    int      j        = 0;
    int      i        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_y);
    __m256   v1       = _mm256_sub_ps(one, v0);
    __m256i  mask32_0 = _mm256_set1_epi32(0x000000FF);
    __m256i  res0, res1, res2, res3;
    for (j = 0; j < dst_width; j += 8)
    {
        __m256i   int8_data_raw0   = _mm256_load_si256((__m256i*)src_row);
        __m256i   int8_data_raw1   = _mm256_load_si256((__m256i*)(src_row + 32));
        __m256i   int8_data_raw2   = _mm256_load_si256((__m256i*)(src_row + 64));
        __m256i   int8_data_raw3   = _mm256_load_si256((__m256i*)(src_row + 96));
        __m256i   cur_int8_data0, cur_int8_data2, cur_int8_data1, cur_int8_data3;
        {
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data2, cur_int8_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_int8_data0   = _mm256_unpacklo_epi32(int8_data_raw0, int8_data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_int8_data2   = _mm256_unpacklo_epi32(int8_data_raw2, int8_data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_int8_data1   = _mm256_unpackhi_epi32(int8_data_raw0, int8_data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_int8_data3   = _mm256_unpackhi_epi32(int8_data_raw2, int8_data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            int8_data_raw0   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            int8_data_raw2   = _mm256_permute2f128_si256(cur_int8_data0, cur_int8_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            int8_data_raw1   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            int8_data_raw3   = _mm256_permute2f128_si256(cur_int8_data1, cur_int8_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }

        __m256    u0                = _mm256_load_ps(pos_x);
        __m256i   cur_int8_data00   = _mm256_and_si256(int8_data_raw0, mask32_0);
        __m256i   cur_int8_data01   = _mm256_and_si256(int8_data_raw1, mask32_0);
        __m256i   cur_int8_data10   = _mm256_and_si256(int8_data_raw2, mask32_0);
        __m256i   cur_int8_data11   = _mm256_and_si256(int8_data_raw3, mask32_0);

        __m256    data0             = _mm256_cvtepi32_ps(cur_int8_data00);
        __m256    data1             = _mm256_cvtepi32_ps(cur_int8_data01);
        __m256    data2             = _mm256_cvtepi32_ps(cur_int8_data10);
        __m256    data3             = _mm256_cvtepi32_ps(cur_int8_data11);
        __m256    u1                = _mm256_sub_ps(one, u0);

        __m256    coef1             = _mm256_mul_ps(u0, v1);
        __m256    coef3             = _mm256_mul_ps(u0, v0);
        __m256    coef0             = _mm256_mul_ps(u1, v1);
        __m256    coef2             = _mm256_mul_ps(u1, v0);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res0                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0,  8), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1,  8), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2,  8), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3,  8), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res1                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 16), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1, 16), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 16), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3, 16), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res2                        = _mm256_cvtps_epi32(data0);

        cur_int8_data00             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw0, 24), mask32_0);
        cur_int8_data01             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw1, 24), mask32_0);
        cur_int8_data10             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw2, 24), mask32_0);
        cur_int8_data11             = _mm256_and_si256(_mm256_srli_epi32(int8_data_raw3, 24), mask32_0);
        data0                       = _mm256_cvtepi32_ps(cur_int8_data00);
        data1                       = _mm256_cvtepi32_ps(cur_int8_data01);
        data2                       = _mm256_cvtepi32_ps(cur_int8_data10);
        data3                       = _mm256_cvtepi32_ps(cur_int8_data11);

        data0                       = _mm256_mul_ps(data0, coef0);
        data1                       = _mm256_mul_ps(data1, coef1);
        data2                       = _mm256_mul_ps(data2, coef2);
        data3                       = _mm256_mul_ps(data3, coef3);
        data0                       = _mm256_add_ps(data0, data1);
        data2                       = _mm256_add_ps(data2, data3);
        data0                       = _mm256_add_ps(data0, data2);
        res3                        = _mm256_cvtps_epi32(data0);

        res0                        = _mm256_min_epi32(res0, mask32_0);
        res1                        = _mm256_min_epi32(res1, mask32_0);
        res2                        = _mm256_min_epi32(res2, mask32_0);
        res3                        = _mm256_min_epi32(res3, mask32_0);
        res1                        = _mm256_slli_epi32(res1, 8);
        res2                        = _mm256_slli_epi32(res2, 16);
        res3                        = _mm256_slli_epi32(res3, 24);
        res0                        = _mm256_or_si256(res0, res1);
        res2                        = _mm256_or_si256(res2, res3);
        res0                        = _mm256_or_si256(res0, res2);

        _mm256_storeu_si256((__m256i*)dst_row, res0);
        src_row                    += 128;
        dst_row                    += 32;
        pos_x                      += 8;
    }
}
#endif

void resize_image_f1_row_proc_implement_avx(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_v);
    __m256   v1       = _mm256_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 8)
    {
        __m256   data_raw0   = _mm256_load_ps(src_row);
        __m256   data_raw1   = _mm256_load_ps((src_row + 8));
        __m256   data_raw2   = _mm256_load_ps((src_row + 16));
        __m256   data_raw3   = _mm256_load_ps((src_row + 24));
        __m256   u0          = _mm256_load_ps(pos_u);
        __m256   cur_data0, cur_data2, cur_data1, cur_data3;
        {
            cur_data0   = _mm256_unpacklo_ps(data_raw0, data_raw1); // 0,  8,  1,  9,  4, 12,  5, 13
            cur_data2   = _mm256_unpacklo_ps(data_raw2, data_raw3); //16, 24, 17, 25, 20, 28, 21, 29
            cur_data1   = _mm256_unpackhi_ps(data_raw0, data_raw1); // 2, 10,  3, 11,  6, 14,  7, 15 
            cur_data3   = _mm256_unpackhi_ps(data_raw2, data_raw3); //18, 26, 19, 27, 22, 30, 23, 31
            data_raw0   = _mm256_permute2f128_ps(cur_data0, cur_data1, 32);  // 0,  8,  1,  9,  2, 10,  3, 11
            data_raw1   = _mm256_permute2f128_ps(cur_data0, cur_data1, 49);  // 4, 12,  5, 13,  6, 14,  7, 15
            data_raw2   = _mm256_permute2f128_ps(cur_data2, cur_data3, 32);  //16, 24, 17, 25, 18, 26, 19, 27
            data_raw3   = _mm256_permute2f128_ps(cur_data2, cur_data3, 49);  //20, 28, 21, 29, 22, 30, 23, 31
            cur_data0   = _mm256_unpacklo_ps(data_raw0, data_raw1);          // 0,  4,  8, 12,  2,  6, 10, 14
            cur_data2   = _mm256_unpacklo_ps(data_raw2, data_raw3);          //16, 20, 24, 28, 18, 22, 26, 30
            cur_data1   = _mm256_unpackhi_ps(data_raw0, data_raw1);          // 1,  5,  9, 13,  3,  7, 11, 15
            cur_data3   = _mm256_unpackhi_ps(data_raw2, data_raw3);          //17, 21, 25, 29, 19, 23, 27, 31
            data_raw0   = _mm256_permute2f128_ps(cur_data0, cur_data2, 32);  // 0,  4,  8, 12, 16, 20, 24, 28
            data_raw2   = _mm256_permute2f128_ps(cur_data0, cur_data2, 49);  // 2,  6, 10, 14, 18, 22, 26, 30
            data_raw1   = _mm256_permute2f128_ps(cur_data1, cur_data3, 32);  // 1,  5,  9, 13, 17, 21, 25, 29
            data_raw3   = _mm256_permute2f128_ps(cur_data1, cur_data3, 49);  // 3,  7, 11, 15, 19, 23, 27, 31
        }
        __m256    u1               = _mm256_sub_ps(one, u0);
        __m256    res1             = _mm256_mul_ps(u0, v1);
        __m256    res3             = _mm256_mul_ps(u0, v0);
        __m256    res0             = _mm256_mul_ps(u1, v1);
        __m256    res2             = _mm256_mul_ps(u1, v0);

        res0                       = _mm256_mul_ps(data_raw0, res0);
        res1                       = _mm256_mul_ps(data_raw1, res1);
        res2                       = _mm256_mul_ps(data_raw2, res2);
        res3                       = _mm256_mul_ps(data_raw3, res3);
        res0                       = _mm256_add_ps(res0, res1);
        res2                       = _mm256_add_ps(res2, res3);
        res0                       = _mm256_add_ps(res0, res2);
        
        _mm256_storeu_ps(dst_row, res0);   
        src_row                   += 32;
        dst_row                   += 8;
        pos_u                     += 8;
    }
}

/*
void resize_image_f1_row_proc_implement_avx(float const*         src_row_0, 
                                            float const*         src_row_1, 
                                            int const*           pos_x_0,
                                            int const*           pos_x_1,
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_v);
    __m256   v1       = _mm256_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 8)
    {
        __m256   data_raw0   = _mm256_set_ps(src_row_0[pos_x_0[7]], src_row_0[pos_x_0[6]], src_row_0[pos_x_0[5]], src_row_0[pos_x_0[4]],
                                             src_row_0[pos_x_0[3]], src_row_0[pos_x_0[2]], src_row_0[pos_x_0[1]], src_row_0[pos_x_0[0]]);
        __m256   data_raw1   = _mm256_set_ps(src_row_0[pos_x_1[7]], src_row_0[pos_x_1[6]], src_row_0[pos_x_1[5]], src_row_0[pos_x_1[4]],
                                             src_row_0[pos_x_1[3]], src_row_0[pos_x_1[2]], src_row_0[pos_x_1[1]], src_row_0[pos_x_1[0]]);
        __m256   data_raw2   = _mm256_set_ps(src_row_1[pos_x_0[7]], src_row_1[pos_x_0[6]], src_row_1[pos_x_0[5]], src_row_1[pos_x_0[4]],
                                             src_row_1[pos_x_0[3]], src_row_1[pos_x_0[2]], src_row_1[pos_x_0[1]], src_row_1[pos_x_0[0]]);
        __m256   data_raw3   = _mm256_set_ps(src_row_1[pos_x_1[7]], src_row_1[pos_x_1[6]], src_row_1[pos_x_1[5]], src_row_1[pos_x_1[4]],
                                             src_row_1[pos_x_1[3]], src_row_1[pos_x_1[2]], src_row_1[pos_x_1[1]], src_row_1[pos_x_1[0]]);
        __m256   u0          = _mm256_load_ps(pos_u);

        __m256    u1               = _mm256_sub_ps(one, u0);
        __m256    res1             = _mm256_mul_ps(u0, v1);
        __m256    res3             = _mm256_mul_ps(u0, v0);
        __m256    res0             = _mm256_mul_ps(u1, v1);
        __m256    res2             = _mm256_mul_ps(u1, v0);

        res0                       = _mm256_mul_ps(data_raw0, res0);
        res1                       = _mm256_mul_ps(data_raw1, res1);
        res2                       = _mm256_mul_ps(data_raw2, res2);
        res3                       = _mm256_mul_ps(data_raw3, res3);
        res0                       = _mm256_add_ps(res0, res1);
        res2                       = _mm256_add_ps(res2, res3);
        res0                       = _mm256_add_ps(res0, res2);
        
        _mm256_storeu_ps(dst_row, res0);   
        dst_row                   += 8;
        pos_x_0                   += 8;
        pos_x_1                   += 8;
        pos_u                     += 8;
    }
}
*/

void resize_image_f3_row_proc_implement_avx(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_v);
    __m256   v1       = _mm256_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 4)
    {
        __m256   data_raw0   = _mm256_load_ps(src_row);
        __m256   data_raw1   = _mm256_load_ps(src_row + 8);
        __m256   data_raw2   = _mm256_load_ps(src_row + 16);
        __m256   data_raw3   = _mm256_load_ps(src_row + 24);
        __m256   data_raw4   = _mm256_load_ps(src_row + 32);
        __m256   data_raw5   = _mm256_load_ps(src_row + 40);
        __m256   data_raw6   = _mm256_load_ps(src_row + 48);
        __m256   data_raw7   = _mm256_load_ps(src_row + 56);
        __m256   u00         = _mm256_insertf128_ps(_mm256_set1_ps(pos_u[0]), _mm_set1_ps(pos_u[1]), 1);
        __m256   u01         = _mm256_insertf128_ps(_mm256_set1_ps(pos_u[2]), _mm_set1_ps(pos_u[3]), 1);
        __m256   cur_data0, cur_data2, cur_data1, cur_data3, cur_data4, cur_data5, cur_data6, cur_data7;
        {
            cur_data0   = _mm256_permute2f128_ps(data_raw0, data_raw2, 32);
            cur_data1   = _mm256_permute2f128_ps(data_raw4, data_raw6, 32);
            cur_data2   = _mm256_permute2f128_ps(data_raw0, data_raw2, 49);
            cur_data3   = _mm256_permute2f128_ps(data_raw4, data_raw6, 49);
            cur_data4   = _mm256_permute2f128_ps(data_raw1, data_raw3, 32);
            cur_data5   = _mm256_permute2f128_ps(data_raw5, data_raw7, 32);
            cur_data6   = _mm256_permute2f128_ps(data_raw1, data_raw3, 49);
            cur_data7   = _mm256_permute2f128_ps(data_raw5, data_raw7, 49);
        }
        __m256    u10              = _mm256_sub_ps(one, u00);
        __m256    u11              = _mm256_sub_ps(one, u01);

        __m256    coef00           = _mm256_mul_ps(u10, v1);
        __m256    coef01           = _mm256_mul_ps(u00, v1);
        __m256    coef02           = _mm256_mul_ps(u10, v0);
        __m256    coef03           = _mm256_mul_ps(u00, v0);

        __m256    coef04           = _mm256_mul_ps(u11, v1);
        __m256    coef05           = _mm256_mul_ps(u01, v1);
        __m256    coef06           = _mm256_mul_ps(u11, v0);
        __m256    coef07           = _mm256_mul_ps(u01, v0);

        coef00                     = _mm256_mul_ps(cur_data0, coef00);
        coef01                     = _mm256_mul_ps(cur_data2, coef01);
        coef02                     = _mm256_mul_ps(cur_data4, coef02);
        coef03                     = _mm256_mul_ps(cur_data6, coef03);

        coef04                     = _mm256_mul_ps(cur_data1, coef04);
        coef05                     = _mm256_mul_ps(cur_data3, coef05);
        coef06                     = _mm256_mul_ps(cur_data5, coef06);
        coef07                     = _mm256_mul_ps(cur_data7, coef07);

        coef00                     = _mm256_add_ps(coef00, coef01);
        coef02                     = _mm256_add_ps(coef02, coef03);
        coef04                     = _mm256_add_ps(coef04, coef05);
        coef06                     = _mm256_add_ps(coef06, coef07);
        coef00                     = _mm256_add_ps(coef00, coef02);
        coef04                     = _mm256_add_ps(coef04, coef06);
        
        _mm_storeu_ps(dst_row,     _mm256_castps256_ps128(coef00));
        _mm_storeu_ps(dst_row + 3, _mm256_castps256_ps128(_mm256_permute2f128_ps(coef00, coef00, 1)));
        _mm_storeu_ps(dst_row + 6, _mm256_castps256_ps128(coef04));
#if defined(__x86_64__) || defined(_M_X64)
        *(long long*)(dst_row + 9) = _mm256_extract_epi64(_mm256_castps_si256(coef04), 2);
#else
        *(int*)(dst_row + 9)       = _mm256_extract_epi32(_mm256_castps_si256(coef04),  4);
        *(int*)(dst_row + 10)      = _mm256_extract_epi32(_mm256_castps_si256(coef04),  5);
#endif
        *(int*)(dst_row + 11)      = _mm256_extract_epi32(_mm256_castps_si256(coef04),  6);
        src_row                   += 64;
        dst_row                   += 12;
        pos_u                     += 4;
    }
}

void resize_image_f4_row_proc_implement_avx(float const*         src_row, 
                                            float*               dst_row, 
                                            int                  dst_width, 
                                            float const*         pos_u, 
                                            float                pos_v)
{
    int      j        = 0;
    __m256   one      = _mm256_set1_ps(1.0f);
    __m256   v0       = _mm256_set1_ps(pos_v);
    __m256   v1       = _mm256_sub_ps(one, v0);
    for (j = 0; j < dst_width; j += 4)
    {
        __m256   data_raw0   = _mm256_load_ps(src_row);
        __m256   data_raw1   = _mm256_load_ps(src_row + 8);
        __m256   data_raw2   = _mm256_load_ps(src_row + 16);
        __m256   data_raw3   = _mm256_load_ps(src_row + 24);
        __m256   data_raw4   = _mm256_load_ps(src_row + 32);
        __m256   data_raw5   = _mm256_load_ps(src_row + 40);
        __m256   data_raw6   = _mm256_load_ps(src_row + 48);
        __m256   data_raw7   = _mm256_load_ps(src_row + 56);
        __m256   u00         = _mm256_insertf128_ps(_mm256_set1_ps(pos_u[0]), _mm_set1_ps(pos_u[1]), 1);
        __m256   u01         = _mm256_insertf128_ps(_mm256_set1_ps(pos_u[2]), _mm_set1_ps(pos_u[3]), 1);
        __m256   cur_data0, cur_data2, cur_data1, cur_data3, cur_data4, cur_data5, cur_data6, cur_data7;
        {
            cur_data0   = _mm256_permute2f128_ps(data_raw0, data_raw2, 32);
            cur_data1   = _mm256_permute2f128_ps(data_raw4, data_raw6, 32);
            cur_data2   = _mm256_permute2f128_ps(data_raw0, data_raw2, 49);
            cur_data3   = _mm256_permute2f128_ps(data_raw4, data_raw6, 49);
            cur_data4   = _mm256_permute2f128_ps(data_raw1, data_raw3, 32);
            cur_data5   = _mm256_permute2f128_ps(data_raw5, data_raw7, 32);
            cur_data6   = _mm256_permute2f128_ps(data_raw1, data_raw3, 49);
            cur_data7   = _mm256_permute2f128_ps(data_raw5, data_raw7, 49);
        }
        __m256    u10              = _mm256_sub_ps(one, u00);
        __m256    u11              = _mm256_sub_ps(one, u01);

        __m256    coef00           = _mm256_mul_ps(u10, v1);
        __m256    coef01           = _mm256_mul_ps(u00, v1);
        __m256    coef02           = _mm256_mul_ps(u10, v0);
        __m256    coef03           = _mm256_mul_ps(u00, v0);

        __m256    coef04           = _mm256_mul_ps(u11, v1);
        __m256    coef05           = _mm256_mul_ps(u01, v1);
        __m256    coef06           = _mm256_mul_ps(u11, v0);
        __m256    coef07           = _mm256_mul_ps(u01, v0);

        coef00                     = _mm256_mul_ps(cur_data0, coef00);
        coef01                     = _mm256_mul_ps(cur_data2, coef01);
        coef02                     = _mm256_mul_ps(cur_data4, coef02);
        coef03                     = _mm256_mul_ps(cur_data6, coef03);

        coef04                     = _mm256_mul_ps(cur_data1, coef04);
        coef05                     = _mm256_mul_ps(cur_data3, coef05);
        coef06                     = _mm256_mul_ps(cur_data5, coef06);
        coef07                     = _mm256_mul_ps(cur_data7, coef07);

        coef00                     = _mm256_add_ps(coef00, coef01);
        coef02                     = _mm256_add_ps(coef02, coef03);
        coef04                     = _mm256_add_ps(coef04, coef05);
        coef06                     = _mm256_add_ps(coef06, coef07);
        coef00                     = _mm256_add_ps(coef00, coef02);
        coef04                     = _mm256_add_ps(coef04, coef06);
        
        _mm256_storeu_ps(dst_row,     coef00);
        _mm256_storeu_ps(dst_row + 8, coef04);
        src_row                   += 64;
        dst_row                   += 16;
        pos_u                     += 4;
    }
}