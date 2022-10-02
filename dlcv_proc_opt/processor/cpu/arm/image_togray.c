#include <arm_neon.h>

static void image_togray_uc3_row_proc(unsigned char const*        src,
                                      unsigned char*              dst,
                                      int                         src_width,
                                      unsigned short int const*   cvt_coef)
{
    int i = 0;
    const uint16x4_t  coef0  = vdup_n_u16(cvt_coef[0]);
    const uint16x4_t  coef1  = vdup_n_u16(cvt_coef[1]);
    const uint16x4_t  coef2  = vdup_n_u16(cvt_coef[2]);
    const uint8x16_t  zero   = vdupq_n_u8(0);
    const uint32x4_t  half   = vdupq_n_u32(16384);
    for(i = 0 ; i < src_width ; i += 16)
    {
        uint16x8_t data00, data01, data10, data11, data20, data21;
        {
            uint8x16x3_t  d = vld3q_u8(src);
            data00          = vmovl_u8(vget_low_u8(d.val[0]));
            data01          = vmovl_u8(vget_high_u8(d.val[0]));
            data10          = vmovl_u8(vget_low_u8(d.val[1]));
            data11          = vmovl_u8(vget_high_u8(d.val[1]));
            data20          = vmovl_u8(vget_low_u8(d.val[2]));
            data21          = vmovl_u8(vget_high_u8(d.val[2]));
        }

        uint32x4_t res00 = vmull_u16(vget_low_u16(data00),  coef0);
        uint32x4_t res01 = vmull_u16(vget_high_u16(data00), coef0);
        res00            = vmlal_u16(res00, vget_low_u16(data10), coef1); 
        res01            = vmlal_u16(res01, vget_high_u16(data10), coef1); 
        res00            = vmlal_u16(res00, vget_low_u16(data20), coef2); 
        res01            = vmlal_u16(res01, vget_high_u16(data20), coef2); 
        res00            = vrshrq_n_u32(res00, 15);
        res01            = vrshrq_n_u32(res01, 15);
        data00           = vcombine_u16(vqmovn_u32(res00), vqmovn_u32(res01));

        res00            = vmull_u16(vget_low_u16(data01),  coef0);
        res01            = vmull_u16(vget_high_u16(data01), coef0);
        res00            = vmlal_u16(res00, vget_low_u16(data11), coef1); 
        res01            = vmlal_u16(res01, vget_high_u16(data11), coef1); 
        res00            = vmlal_u16(res00, vget_low_u16(data21), coef2); 
        res01            = vmlal_u16(res01, vget_high_u16(data21), coef2); 
        res00            = vrshrq_n_u32(res00, 15);
        res01            = vrshrq_n_u32(res01, 15);
        data01           = vcombine_u16(vqmovn_u32(res00), vqmovn_u32(res01));
        uint8x16_t res   = vcombine_u8(vqmovn_u16(data00), vqmovn_u16(data01));

        vst1q_u8(dst, res);

        src += 48;
        dst += 16;
    }
}

void image_togray_uc3_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef)
{
    int i = 0;
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src + i * src_line_size;
        unsigned char*        cur_dst_ptr = dst + i * dst_line_size;
        image_togray_uc3_row_proc(cur_src_ptr, cur_dst_ptr, src_width, cvt_coef);
    }
}


static void image_togray_uc4_row_proc(unsigned char const*        src,
                                      unsigned char*              dst,
                                      int                         src_width,
                                      unsigned short int const*   cvt_coef)
{
    int i = 0;
    const uint16x4_t  coef0  = vdup_n_u16(cvt_coef[0]);
    const uint16x4_t  coef1  = vdup_n_u16(cvt_coef[1]);
    const uint16x4_t  coef2  = vdup_n_u16(cvt_coef[2]);
    const uint8x16_t  zero   = vdupq_n_u8(0);
    const uint32x4_t  half   = vdupq_n_u32(16384);
    for(i = 0 ; i < src_width ; i += 16)
    {
        uint16x8_t data00, data01, data10, data11, data20, data21;
        {
            uint8x16x4_t  d = vld4q_u8(src);
            data00          = vmovl_u8(vget_low_u8(d.val[0]));
            data01          = vmovl_u8(vget_high_u8(d.val[0]));
            data10          = vmovl_u8(vget_low_u8(d.val[1]));
            data11          = vmovl_u8(vget_high_u8(d.val[1]));
            data20          = vmovl_u8(vget_low_u8(d.val[2]));
            data21          = vmovl_u8(vget_high_u8(d.val[2]));
        }

        uint32x4_t res00 = vmull_u16(vget_low_u16(data00),  coef0);
        uint32x4_t res01 = vmull_u16(vget_high_u16(data00), coef0);
        res00            = vmlal_u16(res00, vget_low_u16(data10), coef1); 
        res01            = vmlal_u16(res01, vget_high_u16(data10), coef1); 
        res00            = vmlal_u16(res00, vget_low_u16(data20), coef2); 
        res01            = vmlal_u16(res01, vget_high_u16(data20), coef2); 
        res00            = vrshrq_n_u32(res00, 15);
        res01            = vrshrq_n_u32(res01, 15);
        data00           = vcombine_u16(vqmovn_u32(res00), vqmovn_u32(res01));

        res00            = vmull_u16(vget_low_u16(data01),  coef0);
        res01            = vmull_u16(vget_high_u16(data01), coef0);
        res00            = vmlal_u16(res00, vget_low_u16(data11), coef1); 
        res01            = vmlal_u16(res01, vget_high_u16(data11), coef1); 
        res00            = vmlal_u16(res00, vget_low_u16(data21), coef2); 
        res01            = vmlal_u16(res01, vget_high_u16(data21), coef2); 
        res00            = vrshrq_n_u32(res00, 15);
        res01            = vrshrq_n_u32(res01, 15);
        data01           = vcombine_u16(vqmovn_u32(res00), vqmovn_u32(res01));
        uint8x16_t res   = vcombine_u8(vqmovn_u16(data00), vqmovn_u16(data01));

        vst1q_u8(dst, res);

        src += 64;
        dst += 16;
    }
}

void image_togray_uc4_implement(unsigned char const*        src,
                                unsigned char*              dst,
                                int                         src_width,
                                int                         src_height,
                                int                         src_line_size,
                                int                         dst_line_size,
                                unsigned short int const*   cvt_coef)
{
    int i = 0;
    for(i = 0 ; i < src_height ; i ++)
    {
        unsigned char const*  cur_src_ptr = src + i * src_line_size;
        unsigned char*        cur_dst_ptr = dst + i * dst_line_size;
        image_togray_uc4_row_proc(cur_src_ptr, cur_dst_ptr, src_width, cvt_coef);
    }
}

static void image_togray_f3_row_proc(float const*   src,
                                     float*         dst,
                                     int            src_width,
                                     float const*   cvt_coef)
{
    int i = 0;
    const float32x4_t  coef0  = vdupq_n_f32(cvt_coef[0]);
    const float32x4_t  coef1  = vdupq_n_f32(cvt_coef[1]);
    const float32x4_t  coef2  = vdupq_n_f32(cvt_coef[2]);
    for(i = 0 ; i < src_width ; i += 8)
    {
        float32x4_t data0, data1, data2, data3, data4, data5;
        {
            float32x4x3_t  d0 = vld3q_f32(src);
            float32x4x3_t  d1 = vld3q_f32(src + 12);
            data0            = d0.val[0];
            data1            = d0.val[1];
            data2            = d0.val[2];
            data3            = d1.val[0];
            data4            = d1.val[1];
            data5            = d1.val[2];
        }

#ifdef __aarch64__
        float32x4_t  res0 = vmulq_f32(data0, coef0);
        float32x4_t  res1 = vmulq_f32(data3, coef0);
        res0              = vfmaq_f32(res0, data1, coef1);
        res1              = vfmaq_f32(res1, data4, coef1);
        res0              = vfmaq_f32(res0, data2, coef2);
        res1              = vfmaq_f32(res1, data5, coef2);
#else
        float32x4_t  res0 = vmulq_f32(data0, coef0);
        float32x4_t  res1 = vmulq_f32(data1, coef1);
        float32x4_t  res2 = vmulq_f32(data2, coef2);
        float32x4_t  res3 = vmulq_f32(data3, coef0);
        float32x4_t  res4 = vmulq_f32(data4, coef1);
        float32x4_t  res5 = vmulq_f32(data5, coef2);
        res0              = vaddq_f32(res0, res1);
        res3              = vaddq_f32(res3, res4);
        res0              = vaddq_f32(res0, res2);
        res3              = vaddq_f32(res3, res5);
        res1              = res3;
#endif
        vst1q_f32(dst,     res0);
        vst1q_f32(dst + 4, res1);

        src += 24;
        dst += 8;
    }
}

void image_togray_f3_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef)
{
    int i = 0;
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src + i * src_line_size;
        float*        cur_dst_ptr = dst + i * dst_line_size;
        image_togray_f3_row_proc(cur_src_ptr, cur_dst_ptr, src_width, cvt_coef);
    }
}


static void image_togray_f4_row_proc(float const*   src,
                                     float*         dst,
                                     int            src_width,
                                     float const*   cvt_coef)
{
    int i = 0;
    const float32x4_t  coef0  = vdupq_n_f32(cvt_coef[0]);
    const float32x4_t  coef1  = vdupq_n_f32(cvt_coef[1]);
    const float32x4_t  coef2  = vdupq_n_f32(cvt_coef[2]);
    for(i = 0 ; i < src_width ; i += 8)
    {
        float32x4_t data0, data1, data2, data3, data4, data5;
        {
            float32x4x4_t  d0 = vld4q_f32(src);
            float32x4x4_t  d1 = vld4q_f32(src + 16);
            data0            = d0.val[0];
            data1            = d0.val[1];
            data2            = d0.val[2];
            data3            = d1.val[0];
            data4            = d1.val[1];
            data5            = d1.val[2];
        }

#ifdef __aarch64__
        float32x4_t  res0 = vmulq_f32(data0, coef0);
        float32x4_t  res1 = vmulq_f32(data3, coef0);
        res0              = vfmaq_f32(res0, data1, coef1);
        res1              = vfmaq_f32(res1, data4, coef1);
        res0              = vfmaq_f32(res0, data2, coef2);
        res1              = vfmaq_f32(res1, data5, coef2);
#else
        float32x4_t  res0 = vmulq_f32(data0, coef0);
        float32x4_t  res1 = vmulq_f32(data1, coef1);
        float32x4_t  res2 = vmulq_f32(data2, coef2);
        float32x4_t  res3 = vmulq_f32(data3, coef0);
        float32x4_t  res4 = vmulq_f32(data4, coef1);
        float32x4_t  res5 = vmulq_f32(data5, coef2);
        res0              = vaddq_f32(res0, res1);
        res3              = vaddq_f32(res3, res4);
        res0              = vaddq_f32(res0, res2);
        res3              = vaddq_f32(res3, res5);
        res1              = res3;
#endif
        vst1q_f32(dst,     res0);
        vst1q_f32(dst + 4, res1);

        src += 32;
        dst += 8;
    }
}

void image_togray_f4_implement(float const*   src,
                               float*         dst,
                               int            src_width,
                               int            src_height,
                               int            src_line_size,
                               int            dst_line_size,
                               float const*   cvt_coef)
{
    int i = 0;
    for(i = 0 ; i < src_height ; i ++)
    {
        float const*  cur_src_ptr = src + i * src_line_size;
        float*        cur_dst_ptr = dst + i * dst_line_size;
        image_togray_f4_row_proc(cur_src_ptr, cur_dst_ptr, src_width, cvt_coef);
    }
}