#!/bin/bash

#ndk_name=android-ndk-r13b-linux-x86_64
ndk_name=android-ndk-r16b
opencv_name=OpenCV-3.1.0-android-sdk

ndk_root=/data/hongji/Projects/Android_dev/${ndk_name}
opencv_android_sdk=/data/hongji/Projects/orion_inf/deps/${opencv_name}

workspace=$PWD
docker run --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it \
    -v ${ndk_root}:/${ndk_name}:ro \
    -v ${opencv_android_sdk}:/${opencv_name}:ro \
    -v ${workspace}:/workspace\
    -w /workspace \
    -e NDK_HOME=/${ndk_name}  \
    reg.ainirobot.com/cv/ubuntu:16.04-dlcv-build-env bash
