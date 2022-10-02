#ifndef __DLCV_PROC_OPT_TO_INT8_H__
#define __DLCV_PROC_OPT_TO_INT8_H__

void image_to_int8_f1uc1_implement(float const*      src,
                                   unsigned char*    dst,
                                   int               src_width,
                                   int               src_height,
                                   int               src_line_size,
                                   int               dst_line_size,
                                   float             scale,
                                   float             bias);

void data_to_int8_implement(float const*      src,
                            unsigned char*    dst,
                            int               data_count,
                            float             scale,
                            float             bias);

#endif