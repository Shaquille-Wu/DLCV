#!/bin/bash

##### linux x86-64 gcc
build=build-linux-x86_64-gcc
rm -rf $build && mkdir -p $build 
pushd $build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/x86_64-linux.gcc.toolchain.cmake
make -j8
popd

##### linux x86-64 clang
build=build-linux-x86_64-clang
rm -rf $build && mkdir -p $build
pushd $build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/x86_64-linux.clang.toolchain.cmake
make -j8
popd

##### android armv8
build=build-android-armv8
rm -rf $build && mkdir -p $build 
pushd $build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_STL=gnustl_shared \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-27 \
    -DOPENCV_ANDROID_SDK_ROOT=/OpenCV-3.1.0-android-sdk
make -j8
popd

##### android armv7
build=build-android-armv7
rm -rf $build && mkdir -p $build 
pushd $build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_STL=gnustl_shared \
    -DANDROID_ABI="armeabi-v7a" \
    -DANDROID_PLATFORM=android-22 \
    -DOPENCV_ANDROID_SDK_ROOT=/OpenCV-3.1.0-android-sdk
make -j8
popd


