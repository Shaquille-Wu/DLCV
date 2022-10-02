/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file dlcv_proc_asm_com_def.h
 * @brief This header file defines DLCVAsm struct
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-12-20
 */

#ifndef __DLCV_PROC_ASM_COM_DEF_H__
#define __DLCV_PROC_ASM_COM_DEF_H__
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
#endif
.endm


#endif /* __DLCV_PROC_ASM_COM_DEF_H__ */
