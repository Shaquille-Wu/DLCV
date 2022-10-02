//
//  resize_image_uc1_row_proc_implement.S
//  DLCV
//
//  Created by ORION on 2020/12/28.
//  Copyright © 2020, OrionStar
//

#ifdef __aarch64__

#include "../../dlcv_proc_asm_com_def.h"

.text
.align 5

asm_function resize_image_uc1_row_proc_implement
//void resize_image_uc1_row_proc_implement(unsigned char const*       src_row, 
//                                         unsigned char*             dst_row, 
//                                         int                        dst_width, 
//                                         unsigned short int const*  pos_u,  
//                                         unsigned long long         pos_v1, 
//                                         unsigned long long         pos_v0)
//x0:src_row, x1:dst_row, x2:dst_width, x3:pos_u, x4:pos_v1, x5:pos_v0

mov  w6,        #00FFh            //mask32_flag 
mov  w7,        #0000FFFFh        //mask32_flag_0
dup  v6.4s,     w6                //mask32_flag           
dup  v7.4s,     w7                //mask32_flag_0
mov  w6,        #0020h            //half
lsl  w6,        w6,         #16
dup  v16.4s,    w6                //half
dup  v17.2d,    x4                //pos_v1
dup  v18.2d,    x5                //pos_v0

AND  v17.16b,   v17.16b,   v7.16b
AND  v18.16b,   v18.16b,   v7.16b

cmp  w2, #0
beq FUNCOVER

ROWLOOP:
ld1 {v5.4s},  [x0],  #16        //src
ld1 {v19.4s}, [x3],  #16        //pos_u

AND           v0.16b,     v5.16b,     v6.16b
USHR          v20.4s,     v5.4s,      #8
USHR          v21.4s,     v5.4s,      #16
USHR          v3.4s,      v5.4s,      #24
AND           v1.16b,     v20.16b,    v6.16b
AND           v2.16b,     v21.16b,    v6.16b

AND           v4.16b,     v19.16b,    v7.16b     //u1
USHR          v5.4s,      v19.4s,     #16        //u0

MUL           v0.4s,      v0.4s,     v4.4s
MUL           v1.4s,      v1.4s,     v5.4s
MUL           v2.4s,      v2.4s,     v4.4s
MUL           v3.4s,      v3.4s,     v5.4s

ADD           v0.4s,      v0.4s,     v1.4s
ADD           v2.4s,      v2.4s,     v3.4s

MUL           v0.4s,      v0.4s,     v17.4s
MUL           v2.4s,      v2.4s,     v18.4s

ADD           v0.4s,      v0.4s,     v2.4s
ADD           v0.4s,      v0.4s,     v16.4s

USHR          v0.4s,      v0.4s,     #22
UQXTN         v1.4h,      v0.4s
UQXTN         v2.8b,      v1.8h

st1           {v2.s}[0],  [x1],      #4
subs          w2,         w2,        #4
bne ROWLOOP

FUNCOVER:
ret

#endif
