//
//  data_to_int8_implement.S
//  DLCV
//
//  Created by ORION on 2021/01/23.
//  Copyright © 2021, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5


asm_function data_to_int8_implement
//void data_to_int8_implement(float const* src, unsigned char* dst, int data_count, float scale, float bias)
//x0:src, x1:dst, x2:data_count, s0:scale, s1:bias

dup  v6.4s,    v0.s[0]
dup  v7.4s,    v1.s[0]

eor  v16.16b,   v16.16b,      v16.16b
cmp  w2,        #0
beq FUNCOVER

PROCLOOP:
ld1 { v2.4s,  v3.4s},  [x0], #32
ld1 { v4.4s,  v5.4s},  [x0], #32

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

sqxtun   v2.8b,  v2.8h
sqxtun2  v2.16b, v4.8h

st1     {v2.4s},    [x1], #16

subs     w2,    w2,       #16
bne PROCLOOP

FUNCOVER:
ret

#endif
