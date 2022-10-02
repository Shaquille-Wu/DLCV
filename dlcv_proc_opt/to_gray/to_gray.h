#ifndef __DLCV_PROC_OPT_TO_GRAY_H__
#define __DLCV_PROC_OPT_TO_GRAY_H__

void image_togray_uc3_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef);

void image_togray_uc4_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef);

void image_togray_f3_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef);

void image_togray_f4_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef);

#endif