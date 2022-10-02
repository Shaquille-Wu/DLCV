#ifndef  __ORION_DLC_ERROR_H__
#define  __ORION_DLC_ERROR_H__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detection_out_error.h
 * @brief This header file defines error code for detection out
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-04-08
 */

namespace vision{

typedef enum tag_detection_out_error_code{
    NONE = 0,
    INVALID_JSON_FILE,
    INVALID_MODEL_FILE,
    POST_PROCESS_SETUP_FAILED,
    MODEL_BUILD_FAILED,
    UNKNOWN,
    SUM
}DETECTION_OUT_ERROR_CODE;

static const char*  kDetectionOutError[DETECTION_OUT_ERROR_CODE::SUM] = {
    "OK",
    "invalid json file",
    "invalid model file",
    "post process setup failed",
    "model build failed",
    "unknown error",
};

} //namespace vision


#endif