cmake_minimum_required(VERSION 3.5)
project(dlcv C CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-sign-compare")

if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions -Wall -Wno-unused-variable -Wno-unused-private-field")
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1) # gtest on android need this
    if(ANDROID_STL STREQUAL "gnustl_static" OR ANDROID_STL STREQUAL "gnustl_shared")
        add_definitions(-D_GLIBCXX_USE_C99=1) # set for json.hpp when using gnustl
        if(ANDROID_ABI STREQUAL "armeabi-v7a")
          add_definitions(-D_GLIBCXX_HAVE_WCSTOF=1) # set for json.hpp when using gnustl
        endif()
    endif()
endif()


# cmake tools 
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

# find opencv
if(ANDROID)

  # download from http://10.60.242.21:8000/opencv_3.4.8_cpp_shared_nkd16b.zip
  set(OpenCV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/deps/opencv/include")
  set(OpenCV_LIB_DIRS "${CMAKE_SOURCE_DIR}/deps/opencv/lib/${CMAKE_ANDROID_ARCH_ABI}")
else()
  find_package(OpenCV REQUIRED)
endif()

set(DLCV_PROC_OPT_OBJ_TO_LINK "")
include(${CMAKE_SOURCE_DIR}/dlcv_proc_opt/CMakeLists.txt)

# include directory
include_directories(
    include
    src/common
    src/interface
    src/inference
    src/preprocess
    src/postprocess
    ${OpenCV_INCLUDE_DIRS}
)


file(GLOB_RECURSE COMMON_SRC    "src/common/*.cpp")
file(GLOB_RECURSE PREPROC_SRC   "src/preprocess/*.cpp")
file(GLOB_RECURSE POSTPROC_SRC  "src/postprocess/*.cpp")
file(GLOB_RECURSE INFERENCE_SRC "src/inference/*.cpp")
file(GLOB_RECURSE INTERFACE_SRC "src/interface/*.cpp")

set(LIBDLCV dlcv)
add_library(
    ${LIBDLCV} 
    STATIC 
    ${COMMON_SRC}
    ${PREPROC_SRC}
    ${POSTPROC_SRC}
    ${INFERENCE_SRC}
    ${INTERFACE_SRC}
    ${DLCV_PROC_OPT_OBJ_TO_LINK}
    )
if(ANDROID)
    target_link_libraries(${LIBDLCV} PUBLIC log)
endif()

# print
message("===============================================================================")
string_status("Build type"          ${CMAKE_BUILD_TYPE})
string_status("C compiler"          ${CMAKE_C_COMPILER})
string_status("C++ compiler"        ${CMAKE_CXX_COMPILER})
if(ANDROID)
  string_status("Android ABI"       ${ANDROID_ABI})
  string_status("Android STL"       ${ANDROID_STL})
endif()
string_status("BUILD_TESTING"       ${BUILD_TESTING})
message("===============================================================================")

# Build unittests if BUILD_TESTING is ON
# and HOST is running Linux, macOS or other unix like platforms
if (BUILD_TESTING AND UNIX)
  add_subdirectory(tools) 
  add_subdirectory(test)
  add_subdirectory(demo)
endif ()
