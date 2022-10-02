#ifndef __RESIZE_IMAGE_H__
#define __RESIZE_IMAGE_H__

#include "../common/dlcv_proc_opt_com_def.h"

#define  RESIZE_UC1_PROC_ALIGNED    8
#define  RESIZE_F1_PROC_ALIGNED     8

void resize_image_uc1_implement(const unsigned char*        src,
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
                                unsigned char*              src_row_extract_buf);

void resize_image_uc3_implement(const unsigned char*        src,
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
                                unsigned char*              src_row_extract_buf);

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
                                unsigned int                alpha_flag);
#endif

void resize_image_f1_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf);

void resize_image_f3_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf);

void resize_image_f4_implement(const float*           src,
                               float*                 dst,
                               int                    src_line_size,
                               int                    dst_width_aligned,
                               int                    dst_height,
                               int                    dst_line_size,
                               const float*           pos_x,
                               const int*             pos_x_0_suppress,
                               const int*             pos_x_1_limit,
                               const float*           pos_y,
                               const int*             pos_y_0_suppress,
                               const int*             pos_y_1_suppress,
                               float*                 src_row_extract_buf);