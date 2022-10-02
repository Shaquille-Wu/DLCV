//
//  normalize_image_uc4f4_implement.S
//  DLCV
//
//  Created by ORION on 2020/12/20.
//  Copyright © 2020, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5


asm_function normalize_image_uc4f4_implement
//void normalize_image_uc4f4_implement(const unsigned char* src, float* dst, int width, int height, int src_line_elem, int dst_line_elem, const float* mean, const float* inv_std);
//x0:src, x1:dst, x2:width, x3:height, x4:src_line_elem, x5:dst_line_elem, x6:mean, x7:inv_std

ld1 {v26.4s}, [x6],    #16
ld1 {v27.4s}, [x7],    #16

mov  x8, x2

cmp  x3, #0
beq FUNCOVER

lsl  x5,  x5, 2
mov  x9,  x0
mov  x10, x1

VERTLOOP:
mov  x0,  x9
mov  x1,  x10
mov  x2,  x8
HORLOOP:
ld1  {v16.4s, v17.4s}, [x0],   #32

uxtl         v18.8h,  v16.8b
uxtl2        v16.8h,  v16.16b
uxtl         v19.8h,  v17.8b
uxtl2        v17.8h,  v17.16b

uxtl         v0.4s,   v18.4h
uxtl2        v1.4s,   v18.8h
uxtl         v2.4s,   v16.4h
uxtl2        v3.4s,   v16.8h
uxtl         v4.4s,   v19.4h
uxtl2        v5.4s,   v19.8h
uxtl         v6.4s,   v17.4h
uxtl2        v7.4s,   v17.8h

ucvtf        v0.4s,   v0.4s
ucvtf        v1.4s,   v1.4s
ucvtf        v2.4s,   v2.4s
ucvtf        v3.4s,   v3.4s
ucvtf        v4.4s,   v4.4s
ucvtf        v5.4s,   v5.4s
ucvtf        v6.4s,   v6.4s
ucvtf        v7.4s,   v7.4s

fsub v0.4s,  v0.4s,  v26.4s
fsub v1.4s,  v1.4s,  v26.4s
fsub v2.4s,  v2.4s,  v26.4s
fsub v3.4s,  v3.4s,  v26.4s
fsub v4.4s,  v4.4s,  v26.4s
fsub v5.4s,  v5.4s,  v26.4s
fsub v6.4s,  v6.4s,  v26.4s
fsub v7.4s,  v7.4s,  v26.4s

fmul v0.4s,  v0.4s,  v27.4s
fmul v1.4s,  v1.4s,  v27.4s
fmul v2.4s,  v2.4s,  v27.4s
fmul v3.4s,  v3.4s,  v27.4s
fmul v4.4s,  v4.4s,  v27.4s
fmul v5.4s,  v5.4s,  v27.4s
fmul v6.4s,  v6.4s,  v27.4s
fmul v7.4s,  v7.4s,  v27.4s

st1  {v0.4s, v1.4s}, [x1],   #32
st1  {v2.4s, v3.4s}, [x1],   #32
st1  {v4.4s, v5.4s}, [x1],   #32
st1  {v6.4s, v7.4s}, [x1],   #32

subs x2, x2, #8
bne HORLOOP

add  x9,  x9,  x4
add  x10, x10, x5
subs x3,  x3,  #1
bne VERTLOOP

FUNCOVER:
ret
#endif