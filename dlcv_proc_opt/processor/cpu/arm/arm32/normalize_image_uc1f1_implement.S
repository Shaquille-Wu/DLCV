//
//  normalize_image_uc1f1_implement.S
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

asm_function normalize_image_uc1f1_implement
//void normalize_image_uc1f1_implement(const unsigned char* src, float* dst, int width, int height, int src_line_elem, int dst_line_elem, float mean, float inv_std);
//r0:src, r1:dst, r2:width, r3:height

push  {r4, r5, r6, r7, r8, lr}
vpush {q4-q7}

mov        r4,   r2
ldr        r5,  [sp, #88]    //src_line_elem
ldr        r6,  [sp, #92]    //dst_line_elem
ldr       r12,  [sp, #96]    //mean
vdup.f32   q4,  r12
ldr       r12,  [sp, #100]   //inv_std
vdup.f32   q5,  r12

cmp        r3,  #0
beq FUNCOVER

lsl         r6,  r6,   2
mov         r7,  r0
mov         r8,  r1

VERTLOOP:
mov         r0,  r7
mov         r1,  r8
mov         r2,  r4
HORLOOP:
vld1.8          {d12, d13},     [r0]!
vmovl.u8        q7,  d12
vmovl.u8	    q6,  d13
vmovl.u16	    q0,  d14
vmovl.u16	    q1,  d15
vmovl.u16	    q2,  d12
vmovl.u16	    q3,  d13

vcvt.f32.u32	q0,  q0
vcvt.f32.u32	q1,  q1
vcvt.f32.u32	q2,  q2
vcvt.f32.u32	q3,  q3

vsub.f32        q0, q0, q4
vsub.f32        q1, q1, q4
vsub.f32        q2, q2, q4
vsub.f32        q3, q3, q4

vmul.f32        q0, q0, q5
vmul.f32        q1, q1, q5
vmul.f32        q2, q2, q5
vmul.f32        q3, q3, q5
vst1.32         {q0, q1}, [r1]!
vst1.32         {q2, q3}, [r1]!

subs        r2, r2, #16
bne HORLOOP
add         r7, r7, r5
add         r8, r8, r6
subs        r3, r3, #1
bne VERTLOOP

FUNCOVER:
vpop      {q4-q7}
pop       {r4, r5, r6, r7, r8, pc}
#endif
#endif
