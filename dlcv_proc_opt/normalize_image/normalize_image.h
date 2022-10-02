#ifndef __NORMALIZE_IMAGE_H__
#define __NORMALIZE_IMAGE_H__

#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || defined(__x86_64))
#define DLCV_USE_AVX

#ifdef DLCV_USE_AVX
#define  UC1F1_PROC_ALIGNED  32
#define  UC3F3_PROC_ALIGNED  48
#define  UC4F4_PROC_ALIGNED  64
#define  F1F1_PROC_ALIGNED   32
#define  F3F3_PROC_ALIGNED   48
#define  F4F4_PROC_ALIGNED   64
#else
#define  UC1F1_PROC_ALIGNED  16
#define  UC3F3_PROC_ALIGNED  24
#define  UC4F4_PROC_ALIGNED  32
#define  F1F1_PROC_ALIGNED   16
#define  F3F3_PROC_ALIGNED   24
#define  F4F4_PROC_ALIGNED   32
#endif
#else
#define  UC1F1_PROC_ALIGNED  16
#define  UC3F3_PROC_ALIGNED  24
#define  UC4F4_PROC_ALIGNED  32
#define  F1F1_PROC_ALIGNED   16
#define  F3F3_PROC_ALIGNED   24
#define  F4F4_PROC_ALIGNED   32
#endif

void normalize_image_uc1f1_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     float                  mean, 
                                     float                  inv_std);

void normalize_image_uc3f3_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     const float*           mean, 
                                     const float*           inv_std);

void normalize_image_uc4f4_implement(const unsigned char*   src, 
                                     float*                 dst, 
                                     int                    width_aligned, 
                                     int                    height, 
                                     int                    src_line_element, 
                                     int                    dst_line_element, 
                                     const float*           mean, 
                                     const float*           inv_std);

void normalize_image_f1f1_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    float          mean, 
                                    float          inv_std);

void normalize_image_f3f3_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    const float*   mean, 
                                    const float*   inv_std);

void normalize_image_f4f4_implement(const float*   src, 
                                    float*         dst, 
                                    int            width_aligned, 
                                    int            height, 
                                    int            src_line_element, 
                                    int            dst_line_element, 
                                    const float*   mean, 
                                    const float*   inv_std);
#endif