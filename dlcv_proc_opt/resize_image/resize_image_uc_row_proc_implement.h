#include "resize_image.h"

void resize_image_uc1_row_proc_implement(unsigned char const*         src_row, 
                                         unsigned char*               dst_row, 
                                         int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                         unsigned short int const*    pos_u, 
                                         unsigned long long           pos_v1,
                                         unsigned long long           pos_v0);
#else
                                         float const*                 pos_u, 
                                         float                        pos_v);
#endif

void resize_image_uc3_row_proc_implement(unsigned char const*         src_row, 
                                         unsigned char*               dst_row, 
                                         int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                         unsigned short int const*    pos_u, 
                                         unsigned long long           pos_v1,
                                         unsigned long long           pos_v0);
#else
                                         float const*                 pos_u, 
                                         float                        pos_v);
#endif

void resize_image_uc4_row_proc_alpha_fixed_implement(unsigned char const*         src_row, 
                                                     unsigned char*               dst_row, 
                                                     int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                                     unsigned short int const*    pos_u, 
                                                     unsigned long long           pos_v1,
                                                     unsigned long long           pos_v0,
#else
                                                     float const*                 pos_u, 
                                                     float                        pos_v
#endif
                                                     unsigned char                alpha_value);

void resize_image_uc4_row_proc_alpha_var_implement(unsigned char const*         src_row, 
                                                   unsigned char*               dst_row, 
                                                   int                          dst_width, 
#ifdef RESIZE_UC_USE_FIXED_PT
                                                   unsigned short int const*    pos_u, 
                                                   unsigned long long           pos_v1,
                                                   unsigned long long           pos_v0);
#else
                                                   float const*                 pos_u, 
                                                   float                        pos_v);
#endif