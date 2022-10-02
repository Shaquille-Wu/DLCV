//
//  normalize_image_uc4f4_implement.S
//  DLCV
//
//  Created by ORION on 2020/12/20.
//  Copyright © 2020, OrionStar
//

#ifdef __arm__
#ifndef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5

asm_function normalize_image_uc4f4_implement
//void normalize_image_uc4f4_implement(const unsigned char* src, float* dst, int width, int height, int src_line_elem, int dst_line_elem, const float* mean, const float* inv_std);
//r0:src, r1:dst, r2:width, r3:height

push {r4, r5, r6, r7, r8, lr}
vpush {q4-q9}

mov r4, r2
ldr r5,  [sp, #120]    //src_line_elem
ldr r6,  [sp, #124]    //dst_line_elem
ldr r12, [sp, #128]
vld1.32  {q8}, [r12]!
ldr r12, [sp, #132]
vld1.32  {q9}, [r12]!

cmp r3, #0
beq FUNCOVER

lsl r6, r6, 2
mov  r7, r0
mov  r8, r1

VERTLOOP:
mov  r0, r7
mov  r1, r8
mov  r2, r4
HORLOOP:
vld1.16  {q12, q13}, [r0]!

vmovl.u8        q15,  d27
vmovl.u8        q14,  d26
vmovl.u8	    q13,  d24
vmovl.u8	    q12,  d25

vmovl.u16	    q0,   d26
vmovl.u16	    q1,   d27
vmovl.u16	    q2,   d24
vmovl.u16	    q3,   d25
vmovl.u16	    q4,   d28
vmovl.u16	    q5,   d29
vmovl.u16	    q6,   d30
vmovl.u16	    q7,   d31

vcvt.f32.u32	q0,  q0
vcvt.f32.u32	q1,  q1
vcvt.f32.u32	q2,  q2
vcvt.f32.u32	q3,  q3
vcvt.f32.u32	q4,  q4
vcvt.f32.u32	q5,  q5
vcvt.f32.u32	q6,  q6
vcvt.f32.u32	q7,  q7


vsub.f32 q0, q0, q8
vsub.f32 q1, q1, q8
vsub.f32 q2, q2, q8
vsub.f32 q3, q3, q8
vsub.f32 q4, q4, q8
vsub.f32 q5, q5, q8
vsub.f32 q6, q6, q8
vsub.f32 q7, q7, q8

vmul.f32 q0, q0, q9
vmul.f32 q1, q1, q9
vmul.f32 q2, q2, q9
vmul.f32 q3, q3, q9
vmul.f32 q4, q4, q9
vmul.f32 q5, q5, q9
vmul.f32 q6, q6, q9
vmul.f32 q7, q7, q9

vst1.32 {q0, q1}, [r1]!
vst1.32 {q2, q3}, [r1]!
vst1.32 {q4, q5}, [r1]!
vst1.32 {q6, q7}, [r1]!

subs r2, r2, #8
bne HORLOOP

add  r7, r7, r5
add  r8, r8, r6
subs r3, r3, #1
bne VERTLOOP

FUNCOVER:
vpop {q4-q9}
pop {r4, r5, r6, r7, r8, pc}
#endif
#endif
