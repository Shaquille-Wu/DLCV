cmake_minimum_required(VERSION 3.5)

enable_testing()

if(ANDROID)
  LIST(APPEND OpenCV_LIBRARIES  ${OpenCV_LIB_DIRS}/libopencv_highgui.so   ${OpenCV_LIB_DIRS}/libopencv_imgproc.so     ${OpenCV_LIB_DIRS}/libopencv_core.so     ${OpenCV_LIB_DIRS}/libopencv_imgcodecs.so ${OpenCV_LIB_DIRS}/libopencv_videoio.so)
  string_status("OpenCV_LIBRARIES" ${OpenCV_LIBRARIES})
endif()

# create GTest
add_library(gtest STATIC EXCLUDE_FROM_ALL gtest/gtest.h gtest/gtest-all.cpp)
target_include_directories(gtest PUBLIC "${PROJECT_SOURCE_DIR}/test")
# FIXME: this may break cross-compiling
if (NOT ANDROID)
    target_link_libraries(gtest PUBLIC pthread)
endif()

add_library(gtest_main EXCLUDE_FROM_ALL gtest/gtest_main.cc)
target_link_libraries(gtest_main gtest)

# disable linter for GTest
set_target_properties(gtest gtest_main PROPERTIES CMAKE_CXX_CLANG_TIDY ""
                                                  CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

# disable clang-tidy for gtest framework
foreach(target gtest gtest_main)
  set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "")
endforeach()

add_executable(dlcv_test 
               test_dim.cpp 
               test_blob.cpp 
               test_featuremap_post.cpp 
               test_detector.cpp 
               test_common.cpp 
               test_batch_normalize_image.cpp
               test_normalize_image.cpp 
               test_resize_image.cpp
               test_image_togray.cpp
               test_image_toint8.cpp
               test_superpoint_post_proc.cpp)
target_link_libraries(dlcv_test gtest gtest_main)
target_link_libraries(dlcv_test ${LIBDLCV} ${OpenCV_LIBRARIES} dl)
add_test(NAME Dim COMMAND dlcv_test)
