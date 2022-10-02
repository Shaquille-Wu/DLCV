//
//  normalize_image_f3f3_implement.S
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

asm_function normalize_image_f3f3_implement
//void normalize_image_f3f3_implement(const float* src, float* dst, int width, int height, int src_line_elem, int dst_line_elem, const float* mean, const float* inv_std);
//r0:src, r1:dst, r2:width, r3:height

push {r4, r5, r6, r7, r8, lr}
vpush {q4-q11}

mov r4, r2
ldr r5,  [sp, #152]    //src_line_elem
ldr r6,  [sp, #156]    //dst_line_elem
ldr r12, [sp, #160]
vld1.32  {q6, q7},  [r12]!
vld1.32  {q8},      [r12]!
ldr r12, [sp, #164]
vld1.32  {q9, q10}, [r12]!
vld1.32  {q11},     [r12]!

cmp r3, #0
beq FUNCOVER

lsl r5, r5, 2
lsl r6, r6, 2
mov  r7, r0
mov  r8, r1

VERTLOOP:
mov  r0, r7
mov  r1, r8
mov  r2, r4
HORLOOP:
vld1.32 {q0, q1}, [r0]!
vld1.32 {q2, q3}, [r0]!
vld1.32 {q4, q5}, [r0]!
vsub.f32 q0, q0, q6
vsub.f32 q1, q1, q7
vsub.f32 q2, q2, q8
vsub.f32 q3, q3, q6
vsub.f32 q4, q4, q7
vsub.f32 q5, q5, q8

vmul.f32 q0, q0, q9
vmul.f32 q1, q1, q10
vmul.f32 q2, q2, q11
vmul.f32 q3, q3, q9
vmul.f32 q4, q4, q10
vmul.f32 q5, q5, q11

vst1.32 {q0, q1}, [r1]!
vst1.32 {q2, q3}, [r1]!
vst1.32 {q4, q5}, [r1]!

subs r2, r2, #8
bne HORLOOP

add  r7, r7, r5
add  r8, r8, r6
subs r3, r3, #1
bne VERTLOOP

FUNCOVER:
vpop {q4-q11}
pop {r4, r5, r6, r7, r8, pc}
#endif
#endif
