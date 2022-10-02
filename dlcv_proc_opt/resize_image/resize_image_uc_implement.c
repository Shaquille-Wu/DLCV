
#include "resize_image_uc_row_proc_implement.h"

static void CopyDataWithPosUC1(unsigned char const* src, 
                               unsigned char*       dst, 
                               int                  width, 
                               int                  src_y0, 
                               int                  src_y1, 
                               int const*           pos_x0,
                               int                  pos_x1_valid_start,
                               int                  pos_x1_valid_end)
{
    int j = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        unsigned int x0               = src[src_y0];
        unsigned int x1               = src[src_y1];
        *(unsigned int*)(dst + 4 * j) = (x1 << 24) | (x1 << 16) | (x0 << 8) | x0;
    }
    for(; j <= pos_x1_valid_end ; j ++)
    {
        int          src_x_0 = pos_x0[j];
        unsigned int x0      = *(unsigned short int*)(src + src_y0 + src_x_0);
        unsigned int x1      = *(unsigned short int*)(src + src_y1 + src_x_0);
        *(unsigned int*)(dst + 4 * j)     = (x1 << 16) | x0;
    }

    for(; j < width ; j ++)
    {
        int          src_x_0          = pos_x0[j];
        unsigned int x0               = src[src_y0 + src_x_0];
        unsigned int x1               = src[src_y1 + src_x_0];
        *(unsigned int*)(dst + 4 * j) = (x1 << 24) | (x1 << 16) | (x0 << 8) | x0;
    }
}

static void CopyDataWithPosUC3(unsigned char const* src, 
                               unsigned char*       dst, 
                               int                  width, 
                               int                  src_y0, 
                               int                  src_y1, 
                               int const*           pos_x0,
                               int                  pos_x1_valid_start,
                               int                  pos_x1_valid_end)
{
    int  j       = 0;
    int  cur_pos = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        unsigned int  x00  = *(unsigned short int*)(src + src_y0);
        unsigned int  x01  = *(unsigned char*)(     src + src_y0 + 2);
        unsigned int  x10  = *(unsigned short int*)(src + src_y1);
        unsigned int  x11  = *(unsigned char*)(     src + src_y1 + 2);
        x00 = x00 | (x01 << 16);
        x10 = x10 | (x11 << 16);
        *((unsigned int*)(dst + cur_pos))      = x00;
        *((unsigned int*)(dst + cur_pos +  4)) = x00;
        *((unsigned int*)(dst + cur_pos +  8)) = x10;
        *((unsigned int*)(dst + cur_pos + 12)) = x10;
        cur_pos += 16;
    }

    for(; j <= pos_x1_valid_end - 1; j ++)
    {
        int          src_x_0 = pos_x0[j];
        unsigned int x00     = *(unsigned int*)(src + src_y0 + 3 * src_x_0);
        unsigned int x01     = *(unsigned int*)(src + src_y0 + 3 * src_x_0 + 3);
        unsigned int x10     = *(unsigned int*)(src + src_y1 + 3 * src_x_0);
        unsigned int x11     = *(unsigned int*)(src + src_y1 + 3 * src_x_0 + 3);

        *(unsigned int*)(dst + cur_pos)      = x00;
        *(unsigned int*)(dst + cur_pos +  4) = x01;
        *(unsigned int*)(dst + cur_pos +  8) = x10;
        *(unsigned int*)(dst + cur_pos + 12) = x11;
        cur_pos += 16;
    }

    if(j <= pos_x1_valid_end)
    {
        int          src_x_0 = pos_x0[j];
        unsigned int x00     = *(unsigned int*)(      src + src_y0 + 3 * src_x_0);
        unsigned int x01     = *(unsigned short int*)(src + src_y0 + 3 * src_x_0 + 4);
        unsigned int x10     = *(unsigned int*)(      src + src_y1 + 3 * src_x_0);
        unsigned int x11     = *(unsigned short int*)(src + src_y1 + 3 * src_x_0 + 4);

        *(unsigned int*)(dst + cur_pos)      = x00; //note, there is no neccessary to clear the highest 8 bits
        *(unsigned int*)(dst + cur_pos +  4) = ((x01 << 8) | (x00 >> 24U));
        *(unsigned int*)(dst + cur_pos +  8) = x10; //note, there is no neccessary to clear the highest 8 bits
        *(unsigned int*)(dst + cur_pos + 12) = ((x11 << 8) | (x10 >> 24U));
        cur_pos += 16;
        j++;
    }

    for(; j < width ; j ++)
    {
        int           src_x_0 = pos_x0[j];
        unsigned int  x00     = *(unsigned short int*)(src + src_y0 + 3 * src_x_0);
        unsigned int  x01     = *(unsigned char*)(     src + src_y0 + 3 * src_x_0 + 2);
        unsigned int  x10     = *(unsigned short int*)(src + src_y1 + 3 * src_x_0);
        unsigned int  x11     = *(unsigned char*)(     src + src_y1 + 3 * src_x_0 + 2);
        x00 = x00 | (x01 << 16);
        x10 = x10 | (x11 << 16);
        *((unsigned int*)(dst + cur_pos))      = x00;
        *((unsigned int*)(dst + cur_pos +  4)) = x00;
        *((unsigned int*)(dst + cur_pos +  8)) = x10;
        *((unsigned int*)(dst + cur_pos + 12)) = x10;
        cur_pos += 16;
    }
}

static void CopyDataWithPosUC4(unsigned char const* src, 
                               unsigned char*       dst, 
                               int                  width, 
                               int                  src_y0, 
                               int                  src_y1, 
                               int const*           pos_x0,
                               int                  pos_x1_valid_start,
                               int                  pos_x1_valid_end)
{
    int  j       = 0;
    int  cur_pos = 0;
    for(j = 0 ; j < pos_x1_valid_start ; j ++)
    {
        unsigned int  x00  = *(unsigned int*)(src + src_y0);
        unsigned int  x10  = *(unsigned int*)(src + src_y1);
        *((unsigned int*)(dst + cur_pos))      = x00;
        *((unsigned int*)(dst + cur_pos +  4)) = x00;
        *((unsigned int*)(dst + cur_pos +  8)) = x10;
        *((unsigned int*)(dst + cur_pos + 12)) = x10;
        cur_pos += 16;
    }

    for(; j <= pos_x1_valid_end ; j ++)
    {
        int                src_x_0 = pos_x0[j];
        unsigned long long x00     = *(unsigned long long*)(src + src_y0 + 4 * src_x_0);
        unsigned long long x10     = *(unsigned long long*)(src + src_y1 + 4 * src_x_0);

        *(unsigned long long*)(dst + cur_pos)      = x00;
        *(unsigned long long*)(dst + cur_pos +  8) = x10;
        cur_pos += 16;
    }

    for(; j < width ; j ++)
    {
        int           src_x_0 = pos_x0[j];
        unsigned int  x00     = *(unsigned int*)(src + src_y0 + 4 * src_x_0);
        unsigned int  x10     = *(unsigned int*)(src + src_y1 + 4 * src_x_0);
        *((unsigned int*)(dst + cur_pos))      = x00;
        *((unsigned int*)(dst + cur_pos +  4)) = x00;
        *((unsigned int*)(dst + cur_pos +  8)) = x10;
        *((unsigned int*)(dst + cur_pos + 12)) = x10;
        cur_pos += 16;
    }
}

void resize_image_uc1_implement(const unsigned char*        src,
                                unsigned char*              dst,
                                int                         src_line_size,
                                int                         dst_width_aligned,
                                int                         dst_height,
                                int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_x,
#else 
                                const float*                pos_x,
#endif
                                const int*                  pos_x_0_suppress,
                                const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_y,
#else
                                const float*                pos_y,
#endif
                                const int*                  pos_y_0_suppress,
                                const int*                  pos_y_1_suppress,
                                unsigned char*              src_row_extract_buf)
{
    int    i                   = 0;
    for (i = 0; i < dst_height; i++)
    {
        int                src_y_start0            = pos_y_0_suppress[i] * src_line_size;
        int                src_y_start1            = pos_y_1_suppress[i] * src_line_size;
        unsigned char*     src_row_extract_buf_ptr = src_row_extract_buf;
        unsigned char*     dst_ptr                 = dst + i * dst_line_size;
#ifdef RESIZE_UC_USE_FIXED_PT
        unsigned long long pos_v1                  = ((unsigned long long)pos_y[2 * i])       |
                                                     ((unsigned long long)pos_y[2 * i] << 16) |
                                                     ((unsigned long long)pos_y[2 * i] << 32) |
                                                     ((unsigned long long)pos_y[2 * i] << 48);
        unsigned long long pos_v0                  = ((unsigned long long)pos_y[2 * i + 1])       |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 16) |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 32) |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 48);
#endif
        CopyDataWithPosUC1(src, 
                           src_row_extract_buf_ptr, 
                           dst_width_aligned, 
                           src_y_start0, 
                           src_y_start1, 
                           pos_x_0_suppress, 
                           pos_x_1_limit[0],
                           pos_x_1_limit[1]);
        resize_image_uc1_row_proc_implement(src_row_extract_buf_ptr,
                                            dst_ptr,
                                            dst_width_aligned,
                                            pos_x,
#ifdef RESIZE_UC_USE_FIXED_PT
                                            pos_v1,
                                            pos_v0);
#else
                                            pos_y[i]);
#endif
    }
}

void resize_image_uc3_implement(const unsigned char*        src,
                                unsigned char*              dst,
                                int                         src_line_size,
                                int                         dst_width_aligned,
                                int                         dst_height,
                                int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_x,
#else 
                                const float*                pos_x,
#endif
                                const int*                  pos_x_0_suppress,
                                const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_y,
#else
                                const float*                pos_y,
#endif
                                const int*                  pos_y_0_suppress,
                                const int*                  pos_y_1_suppress,
                                unsigned char*              src_row_extract_buf)
{
    int    i                   = 0;
    for (i = 0; i < dst_height; i++)
    {
        int             src_y_start0            = pos_y_0_suppress[i] * src_line_size;
        int             src_y_start1            = pos_y_1_suppress[i] * src_line_size;
        unsigned char*  src_row_extract_buf_ptr = src_row_extract_buf;
        unsigned char*  dst_ptr                 = dst + i * dst_line_size;
#ifdef RESIZE_UC_USE_FIXED_PT
        unsigned long long pos_v1               = ((unsigned long long)pos_y[2 * i])      |
                                                     ((unsigned long long)pos_y[2 * i] << 16) |
                                                     ((unsigned long long)pos_y[2 * i] << 32) |
                                                     ((unsigned long long)pos_y[2 * i] << 48);
        unsigned long long pos_v0               = ((unsigned long long)pos_y[2 * i + 1])      |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 16) |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 32) |
                                                     ((unsigned long long)pos_y[2 * i + 1] << 48);
#endif
        CopyDataWithPosUC3(src, 
                           src_row_extract_buf_ptr, 
                           dst_width_aligned, 
                           src_y_start0, 
                           src_y_start1, 
                           pos_x_0_suppress, 
                           pos_x_1_limit[0],
                           pos_x_1_limit[1]);
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
        resize_image_uc3_row_proc_implement(src_row_extract_buf_ptr,
                                            dst_ptr,
                                            dst_width_aligned,
                                            pos_x,
#ifdef RESIZE_UC_USE_FIXED_PT
                                            pos_v1,
                                            pos_v0);
#else
                                            pos_y[i]);
#endif
#endif
    }
}

void resize_image_uc4_implement(const unsigned char*        src,
                                unsigned char*              dst,
                                int                         src_line_size,
                                int                         dst_width_aligned,
                                int                         dst_height,
                                int                         dst_line_size,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_x,
#else
                                const unsigned float*       pos_x,
#endif
                                const int*                  pos_x_0_suppress,
                                const int*                  pos_x_1_limit,
#ifdef RESIZE_UC_USE_FIXED_PT
                                const unsigned short int*   pos_y,
#else
                                const float*                pos_y,
#endif
                                const int*                  pos_y_0_suppress,
                                const int*                  pos_y_1_suppress,
                                unsigned char*              src_row_extract_buf,
                                unsigned int                alpha_flag)
{
    int    i                   = 0;
    if(alpha_flag <= 255)
    {
        for (i = 0; i < dst_height; i++)
        {
            int             src_y_start0            = pos_y_0_suppress[i] * src_line_size;
            int             src_y_start1            = pos_y_1_suppress[i] * src_line_size;
            unsigned char*  src_row_extract_buf_ptr = src_row_extract_buf;
            unsigned char*  dst_ptr                 = dst + i * dst_line_size;
#ifdef RESIZE_UC_USE_FIXED_PT
            unsigned long long pos_v1               = ((unsigned long long)pos_y[2 * i])      |
                                                      ((unsigned long long)pos_y[2 * i] << 16) |
                                                      ((unsigned long long)pos_y[2 * i] << 32) |
                                                      ((unsigned long long)pos_y[2 * i] << 48);
            unsigned long long pos_v0               = ((unsigned long long)pos_y[2 * i + 1])      |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 16) |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 32) |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 48);
#endif
            CopyDataWithPosUC4(src, 
                               src_row_extract_buf_ptr, 
                               dst_width_aligned, 
                               src_y_start0, 
                               src_y_start1, 
                               pos_x_0_suppress, 
                               pos_x_1_limit[0],
                               pos_x_1_limit[1]);
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
            resize_image_uc4_row_proc_alpha_fixed_implement(src_row_extract_buf_ptr,
                                                            dst_ptr,
                                                            dst_width_aligned,
                                                            pos_x,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                            pos_v1,
                                                            pos_v0,
#else
                                                            pos_y[i],
#endif
                                                            (unsigned char)alpha_flag);
#endif
        }
    }
    else
    {
        for (i = 0; i < dst_height; i++)
        {
            int             src_y_start0            = pos_y_0_suppress[i] * src_line_size;
            int             src_y_start1            = pos_y_1_suppress[i] * src_line_size;
            unsigned char*  src_row_extract_buf_ptr = src_row_extract_buf;
            unsigned char*  dst_ptr                 = dst + i * dst_line_size;
#ifdef RESIZE_UC_USE_FIXED_PT
            unsigned long long pos_v1               = ((unsigned long long)pos_y[2 * i])       |
                                                      ((unsigned long long)pos_y[2 * i] << 16) |
                                                      ((unsigned long long)pos_y[2 * i] << 32) |
                                                      ((unsigned long long)pos_y[2 * i] << 48);
            unsigned long long pos_v0               = ((unsigned long long)pos_y[2 * i + 1])       |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 16) |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 32) |
                                                      ((unsigned long long)pos_y[2 * i + 1] << 48);
#endif
            CopyDataWithPosUC4(src, 
                               src_row_extract_buf_ptr, 
                               dst_width_aligned, 
                               src_y_start0, 
                               src_y_start1, 
                               pos_x_0_suppress, 
                               pos_x_1_limit[0],
                               pos_x_1_limit[1]);
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
            resize_image_uc4_row_proc_alpha_var_implement(src_row_extract_buf_ptr,
                                                          dst_ptr,
                                                          dst_width_aligned,
                                                          pos_x,
#ifdef RESIZE_UC_USE_FIXED_PT
                                                          pos_v1,
                                                          pos_v0);
#else
                                                          pos_y[i]);
#endif
#endif
        }
    }
}