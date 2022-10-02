//
//  normalize_image_f1f1_implement.S
//  DLCV
//
//  Created by ORION on 2020/12/20.
//  Copyright © 2020, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5

asm_function normalize_image_f1f1_implement
//void normalize_image_f1f1_implement(const float* src, float* dst, int width, int height, int src_line_elem, int dst_line_elem, float mean, float inv_std);
//x0:src, x1:dst, x2:width, x3:height, x4:src_line_elem, x5:dst_line_elem, s0:mean, s1:inv_std

mov  w6,       w2
dup  v6.4s,    v0.s[0]
dup  v7.4s,    v1.s[0]

cmp  w3, #0
beq FUNCOVER

lsl  x4,  x4, 2
lsl  x5,  x5, 2
mov  x7,  x0
mov  x8,  x1

VERTLOOP:
mov  x0, x7
mov  x1, x8
mov  w2, w6
HORLOOP:
ld1 {v2.4s, v3.4s},  [x0], #32
ld1 {v4.4s, v5.4s},  [x0], #32
fsub v2.4s,  v2.4s,  v6.4s
fsub v3.4s,  v3.4s,  v6.4s
fsub v4.4s,  v4.4s,  v6.4s
fsub v5.4s,  v5.4s,  v6.4s

fmul v2.4s,  v2.4s,  v7.4s
fmul v3.4s,  v3.4s,  v7.4s
fmul v4.4s,  v4.4s,  v7.4s
fmul v5.4s,  v5.4s,  v7.4s
st1  {v2.4s, v3.4s}, [x1], #32
st1  {v4.4s, v5.4s}, [x1], #32

subs w2, w2, #16
bne HORLOOP

add  x7,  x7,  x4
add  x8,  x8,  x5
subs w3,  w3,  #1
bne VERTLOOP

FUNCOVER:
ret

#endif
