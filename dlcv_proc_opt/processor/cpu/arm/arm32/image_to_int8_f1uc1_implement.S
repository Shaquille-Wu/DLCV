//
//  image_to_int8_f1uc1_implement.S
//  DLCV
//
//  Created by ORION on 2021/01/23.
//  Copyright © 2021, OrionStar
//

#ifdef __arm__
#ifndef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5

asm_function image_to_int8_f1uc1_implement
//void image_to_int8_f1uc1_implement(float const* src, unsigned char* dst, int src_width, int src_height, int src_line_size, int dst_line_size, float scale, float bias)
//r0:src, r1:dst, r2:src_width, r3:src_height

push  {r4, r5, r6, r7, r8, lr}
vpush {q4, q5, q6}

vmov.f32   q10, #0.5

mov        r4,  #0
vdup.f32   q6,  r4

ldr        r4,  [sp, #72]    //src_line_size
ldr        r5,  [sp, #76]    //dst_line_size
ldr        r6,  [sp, #80]    //scale
vdup.f32   q4,  r6
ldr        r7,  [sp, #84]    //bias
vdup.f32   q5,  r7

cmp        r3,  #0
beq FUNCOVER

lsl        r4,      r4,      2
mov        r6,      r0
mov        r7,      r1
mov        r8,      r2
VERTLOOP:

mov        r0,      r6
mov        r1,      r7
mov        r2,      r8
cmp        r2,      #0
HORLOOP:
vld1.32   {q0, q1}, [r0]!
vld1.32   {q2, q3}, [r0]!

vmul.f32  q0, q0, q4
vmul.f32  q1, q1, q4
vmul.f32  q2, q2, q4
vmul.f32  q3, q3, q4

vadd.f32  q0, q0, q5
vadd.f32  q1, q1, q5
vadd.f32  q2, q2, q5
vadd.f32  q3, q3, q5

vmax.f32  q0, q0, q6
vmax.f32  q1, q1, q6
vmax.f32  q2, q2, q6
vmax.f32  q3, q3, q6

vadd.f32  q0, q0, q10
vadd.f32  q1, q1, q10
vadd.f32  q2, q2, q10
vadd.f32  q3, q3, q10

vcvt.s32.f32 q0, q0
vcvt.s32.f32 q1, q1
vcvt.s32.f32 q2, q2
vcvt.s32.f32 q3, q3

vqmovn.s32  d0, q0
vqmovn.s32  d1, q1
vqmovn.s32  d2, q2
vqmovn.s32  d3, q3

vqmovun.s16 d0, q0
vqmovun.s16 d1, q1

vst1.32   {q0}, [r1]!

subs      r2,      r2,     #16
bne HORLOOP

add       r6,      r6,     r4
add       r7,      r7,     r5
subs      r3,      r3,     #1
bne VERTLOOP

FUNCOVER:
vpop      {q4, q5, q6}
pop       {r4, r5, r6, r7, r8, pc}
#endif
#endif
