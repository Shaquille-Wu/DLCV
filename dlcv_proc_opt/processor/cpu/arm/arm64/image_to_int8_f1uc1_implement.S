//
//  image_to_int8_f1uc1_implement.S
//  DLCV
//
//  Created by ORION on 2021/01/23.
//  Copyright © 2021, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5


asm_function image_to_int8_f1uc1_implement
//void image_to_int8_f1uc1_implement(float const* src, unsigned char* dst, int src_width, int src_height, int src_line_size, int dst_line_size, float scale, float bias)
//x0:src, x1:dst, x2:src_width, x3:src_height, x4: src_line_size, x5: dst_line_size, s0: scale, s1: bias

mov      w6,      w2
dup      v6.4s,   v0.s[0]
dup      v7.4s,   v1.s[0]

eor      v16.16b, v16.16b, v16.16b
cmp      w3,      #0
beq FUNCOVER

lsl      x4,      x4,      2
mov      x7,      x0
mov      x8,      x1
VERTLOOP:
mov      x0,      x7
mov      x1,      x8
mov      w2,      w6
HORLOOP:
ld1     { v2.4s,  v3.4s},  [x0], #32
ld1     { v4.4s,  v5.4s},  [x0], #32

fmul     v2.4s,   v2.4s,  v6.4s
fmul     v3.4s,   v3.4s,  v6.4s
fmul     v4.4s,   v4.4s,  v6.4s
fmul     v5.4s,   v5.4s,  v6.4s

fadd     v2.4s,   v2.4s,  v7.4s
fadd     v3.4s,   v3.4s,  v7.4s
fadd     v4.4s,   v4.4s,  v7.4s
fadd     v5.4s,   v5.4s,  v7.4s

fmax     v2.4s,   v2.4s,  v16.4s
fmax     v3.4s,   v3.4s,  v16.4s
fmax     v4.4s,   v4.4s,  v16.4s
fmax     v5.4s,   v5.4s,  v16.4s

fcvtas   v2.4s,   v2.4s
fcvtas   v3.4s,   v3.4s
fcvtas   v4.4s,   v4.4s
fcvtas   v5.4s,   v5.4s

sqxtn    v2.4h,   v2.4s
sqxtn2   v2.8h,   v3.4s
sqxtn    v4.4h,   v4.4s
sqxtn2   v4.8h,   v5.4s

sqxtun   v2.8b,   v2.8h
sqxtun2  v2.16b,  v4.8h

st1      {v2.4s}, [x1],   #16

subs     w2,      w2,     #16
bne HORLOOP

add      x7,      x7,     x4
add      x8,      x8,     x5
subs     w3,      w3,     #1
bne VERTLOOP

FUNCOVER:
ret

#endif
