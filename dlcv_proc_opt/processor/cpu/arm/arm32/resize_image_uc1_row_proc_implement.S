//
//  resize_image_uc1_row_proc_implement.S
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

asm_function resize_image_uc1_row_proc_implement
//void resize_image_uc1_row_proc_implement(unsigned char const*       src_row, 
//                                         unsigned char*             dst_row, 
//                                         int                        dst_width, 
//                                         unsigned short int const*  pos_u,  
//                                         unsigned long long         pos_v1, 
//                                         unsigned long long         pos_v0);
//r0:src_row, r1:dst_row, r2:dst_width, r3:pos_u

push {r4, r5, r6, r7, lr}
vpush {q4-q11}

ldr          r4,       [sp, #148]         //v1, low 32bit
ldr          r5,       [sp, #156]         //v0, low 32bit

mov          r6,       #00FFh             //mask32_flag 
mov          r7,       #0000FFFFh         //mask32_flag_0
vdup.s32     q7,       r6                 //mask32_flag 
vdup.s32     q8,       r7                 //mask32_flag_0
mov          r6,       #0020h             //half
lsl          r6,       r6,         #16
vdup.s32    q10,       r4                //pos_v1
vdup.s32    q11,       r5                //pos_v0
vdup.s32     q9,       r6                //half

vand        q10,       q10,        q8    //pos_v1
vand        q11,       q11,        q8    //pos_v0

cmp         r2,        #0
beq FUNCOVER

ROWLOOP:
vld1.16       {q15},     [r0]!
vld1.16       {q14},     [r3]!


vshr.U32       q13,     q14,        #16    //u0
vand           q14,     q14,         q8    //u1

vand            q0,     q15,         q7    //x0
vshr.U32        q1,     q15,        #8
vshr.U32        q2,     q15,        #16
vshr.U32        q3,     q15,        #24    //x3
vand            q1,     q1,          q7    //x1
vand            q2,     q2,          q7    //x2

vmul.I32        q0,     q0,          q14
vmul.I32        q1,     q1,          q13
vmul.I32        q2,     q2,          q14
vmul.I32        q3,     q3,          q13

vadd.I32        q0,     q0,          q1
vadd.I32        q2,     q2,          q3

vmul.I32        q0,     q0,          q10
vmul.I32        q2,     q2,          q11

vadd.I32        q0,     q0,          q2
vadd.I32        q0,     q0,          q9
vshr.U32        q0,     q0,         #22

vqmovn.U32      d0,     q0
vqmovn.U16      d0,     q0
vst1.s32        {d0[0]},  [r1]!

subs r2, r2, #4
bne ROWLOOP

FUNCOVER:
vpop {q4-q11}
pop {r4, r5, r6, r7, pc}
#endif
#endif
