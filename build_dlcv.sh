#BUILD_CMD: build, clean, test
BUILD_CMD=build
#BUILD_TYPE: Debug, Release
BUILD_TYPE=Debug
#TARGET_ARCH: x86 x86_64 armv7 armv8
TARGET_ARCH=x86_64
#TARGET_OS: linux android os macos
TARGET_OS=linux
#BUILD_PLATFORM: aarch64-android, arm-android, aarch64-linux, arm-linux, x86_64-linux
#BUILD_PLATFORM=x86_64-linux
#ANDROID_NDK_DIR
#ANDROID_NDK_DIR=/home/shaquille/Android/Sdk/ndk-bundle
#ANDROID_NDK_DIR=/home/shaquille/android-ndk-r16b
ANDROID_NDK_DIR=/home/shaquille/android-ndk-r17c
ANDROID_OPENCV_DIR_PREFIX=/home/shaquille/WorkSpace/OpenCV-android-sdk/OpenCV-4.2.0-android-sdk/sdk/native/jni
#ANDROID_OPENCV_DIR=/home/shaquille/WorkSpace/OpenCV-android-sdk/OpenCV-3.1.0-android-sdk/sdk/native/jni
#OPENCV_ANDROID_SDK_ROOT=/home/shaquille/WorkSpace/OpenCV-android-sdk/OpenCV-3.1.0-android-sdk

PROJECT_NAME=dlcv

CORE_COUNT=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "CORE_COUNT ${CORE_COUNT}"

while getopts ":c:t:a:o:q" opt
do
    case $opt in
        c)
        BUILD_CMD=$OPTARG
        ;;    
        t)
        BUILD_TYPE=$OPTARG
        ;;
        a)
        TARGET_ARCH=$OPTARG
        ;;
        o)
        TARGET_OS=$OPTARG
        ;;        
        ?)
        echo "unknow parameter: $opt"
        exit 1;;
    esac
done

echo "Build Command: ${BUILD_CMD}"

CUR_CMAKE_TOOLCHAIN_FILE=${PWD}/dlcv/cmake/x86_64-linux.gcc.toolchain.cmake

if [ "$TARGET_OS" == "android" ] ; then
    if [ "$TARGET_ARCH" == "armv7" ] ; then
        BUILD_PLATFORM=arm-android
    else
        BUILD_PLATFORM=aarch64-android
    fi
elif [ "$TARGET_OS" == "linux" ] ; then
    if [ "$TARGET_ARCH" == "x86" ] ; then
        BUILD_PLATFORM=x86-linux
    elif [ "$TARGET_ARCH" == "x86_64" ] ; then
        BUILD_PLATFORM=x86_64-linux
    elif [ "$TARGET_ARCH" == "armv7" ] ; then
        BUILD_PLATFORM=arm-linux
    elif [ "$TARGET_ARCH" == "armv8" ] ; then
        BUILD_PLATFORM=aarch64-linux       
    else
         echo "unknown linux arch"
    fi    
else
    echo "unknown OS"
fi

CUR_DIR_PATH=${PWD}
mkdir -p ./build/${BUILD_TYPE}/${BUILD_PLATFORM}/${PROJECT_NAME}
cd ./build/${BUILD_TYPE}/${BUILD_PLATFORM}/${PROJECT_NAME}
echo entring ${PWD} for ${BUILD_CMD} ${PROJECT_NAME} start
BUILD_CMD_LINE="-DCMAKE_INSTALL_PREFIX=${PWD}/install -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DARCH=${TARGET_ARCH} -DOS=${TARGET_OS} ../../../../${PROJECT_NAME}"
if [ "$TARGET_OS" == "android" ] ; then
    if [ "$TARGET_ARCH" == "armv8" ] ; then
        ANDROID_ABI_FORMAT="arm64-v8a"
        ANDROID_PLATFORM=android-27
        ANDROID_OPENCV_DIR=${ANDROID_OPENCV_DIR_PREFIX}/abi-arm64-v8a
    else
        ANDROID_ABI_FORMAT="armeabi-v7a"
        ANDROID_PLATFORM=android-22
        ANDROID_OPENCV_DIR=${ANDROID_OPENCV_DIR_PREFIX}/abi-armeabi-v7a
    fi
    BUILD_CMD_LINE="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_DIR}/build/cmake/android.toolchain.cmake -DANDROID_ABI=${ANDROID_ABI_FORMAT} -DANDROID_PLATFORM=${ANDROID_PLATFORM} -DOpenCV_DIR=${ANDROID_OPENCV_DIR} -DANDROID_STL=c++_shared ${BUILD_CMD_LINE}"
    #BUILD_CMD_LINE="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_DIR}/build/cmake/android.toolchain.cmake -DANDROID_ABI=${ANDROID_ABI_FORMAT} -DANDROID_PLATFORM=${ANDROID_PLATFORM} -DANDROID_STL=c++_shared ${BUILD_CMD_LINE}"
elif [ "$TARGET_ARCH" == "armv7" ] || [ "$TARGET_ARCH" == "armv8" ] ; then
    BUILD_CMD_LINE="-DCMAKE_TOOLCHAIN_FILE=./toolchains/arm-linux-gnueabi.toolchain.cmake ${BUILD_CMD_LINE}"
elif [ "$TARGET_ARCH" == "x86_64" ] && [ "$TARGET_OS" == "linux" ] ; then
    #BUILD_CMD_LINE="-DCMAKE_TOOLCHAIN_FILE=${CUR_CMAKE_TOOLCHAIN_FILE}  ${BUILD_CMD_LINE}"
    BUILD_CMD_LINE="${BUILD_CMD_LINE} -DBUILD_TESTING=1 -DLINUX=1"
    echo "x86_64-linux platform"
else
    echo "unknown platform"
fi

if [ "$BUILD_CMD" == "build" ]; then
    echo  "cmake ${BUILD_CMD_LINE}"
    cmake ${BUILD_CMD_LINE}
    make -j${CORE_COUNT}
    make install
elif 
    [ "$BUILD_CMD" == "clean" ]; then
    make clean
    rm -rf ./tests/CMakeFiles
    rm -rf ./tests/cmake_install.cmake
    rm -rf ./tests/Makefile
    rm -rf ./tools/CMakeFiles
    rm -rf ./tools/cmake_install.cmake
    rm -rf ./tools/Makefile    
    rm -rf ./demo/CMakeFiles
    rm -rf ./demo/cmake_install.cmake
    rm -rf ./demo/Makefile       
    rm -rf ./CMakeFiles
    rm -rf ./CMakeCache.txt
    rm -rf ./cmake_install.cmake
    rm -rf ./CTestTestfile.cmake
    rm -rf ./install_manifest.txt
    rm -rf ./Makefile
elif 
    [ "$BUILD_CMD" == "test" ]; then
    make test
else
    echo "unknown cmd"
fi

cd ${CUR_DIR_PATH}
echo leaving ${PWD} for ${BUILD_CMD} ${PROJECT_NAME} end