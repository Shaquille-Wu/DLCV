//
//  super_point_extract_flag_implement_test.S
//  DLCV
//
//  Created by ORION on 2020/12/28.
//  Copyright © 2020, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5

asm_function super_point_extract_flag_implement_test
//void super_point_extract_flag_implement_test(float const*   prob, 
//                                             int            channel, 
//                                             int*           pt_flag,
//                                             float          threshold)
//x0:prob, x1:channel, x2:pt_flag, s0:threshold

fmov          w3,        s0
dup           v6.4s,     w3                //threshold
mov           w3,        #1                //1    
dup           v5.4s,     w3                //1
mov           w4,        #0                //valid_cnt
mov           x5,        x1                //channel
lsr           w5,        w5,        2
lsl           w5,        w5,        2      //channel_aligned
mov           w8,        #0
sub           w6,        w1,        w5

mov           x1,        x0
cmp           w5,        #0
beq ROWLOOPOVER

ROWLOOP:
ld1           {v0.4s},    [x0],  #16        //prob

FCMGE	      v0.4s,      v0.4s,      v6.4s
AND           v0.16b,     v0.16b,     v5.16b
 
str	          w8,         [x2, w4, uxtw #2]
fmov          w3,         s0
add           w4,         w4,                  w3
add           w8,         w8,                  #1
str	          w8,         [x2, w4, uxtw #2]
mov           w9,         v0.s[1]
add           w4,         w4,                  w9
add           w8,         w8,                  #1
str	          w8,         [x2, w4, uxtw #2]
mov          w10,         v0.s[2]
add           w4,         w4,                 w10
add           w8,         w8,                  #1
str	          w8,         [x2, w4, uxtw #2]
mov          w11,         v0.s[3]
add           w4,         w4,                 w11
add           w8,         w8,                  #1
subs          w5,         w5,                  #4
bne ROWLOOP

ROWLOOPOVER:
cmp           w6,        #0
beq FUNCOVER

TAILLOOP:
ldr	          s7,        [x1, x5, lsl  #2]
str           w5,        [x2, w4, uxtw #2]
fcmp	      s7,        s6
cset	      w11,       ge
add           w4,        w4,         w11
add           w5,        w5,         #1
subs          w6,        w6,         #1
bne TAILLOOP

FUNCOVER:
mov           w0,        w4
ret

#endif
