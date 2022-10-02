cd /data/dlcv/build/
export LD_LIBRARY_PATH=$PWD:$PWD/builds/inference/snpe/install/lib/:$PWD/builds/inference/snpe/install/lib/aarch64-android-clang6.0/:$PWD/opencv/lib/arm64-v8a/:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/dlcv/build/builds/inference/snpe/install/lib/aarch64-android-clang6.0;/data/dlcv/build/builds/inference/snpe/install/lib/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
./dlcv_test --gtest_filter=*.snpe_1_36