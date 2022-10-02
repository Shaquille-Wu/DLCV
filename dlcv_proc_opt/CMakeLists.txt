cmake_minimum_required(VERSION 3.0)
# complier options
enable_language(C)
enable_language(ASM)
set(CMAKE_C_STANDARD 99)
set(CMAKE_VERBOSE_MAKEFILE  ON)
set(DLCV_PROC_OPT dlcv_proc_opt)

# target options
message(STATUS "\tSystem:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        add_definitions(-mfloat-abi=softfp -mfpu=neon)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)    #please define in project/cross-compile/arm.toolchain.cmake
    endif()
endif()


IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

if (NOT (MSVC OR WIN32))
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fstrict-aliasing -ffunction-sections -fdata-sections ")
endif()

FILE(GLOB DLCV_PROC_AArch32_SRC ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/arm/arm32/*.S)
FILE(GLOB DLCV_PROC_AArch64_SRC ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/arm/arm64/*.S)
FILE(GLOB DLCV_PROC_ARM_C_SRC   ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/arm/*.c)

set(DLCV_PROC_OPT_SRC 
    ${CMAKE_CURRENT_LIST_DIR}/normalize_image/normalize_image.c
    ${CMAKE_CURRENT_LIST_DIR}/resize_image/resize_image.c
    ${CMAKE_CURRENT_LIST_DIR}/super_point/super_point.c
    ${CMAKE_CURRENT_LIST_DIR}/to_gray/to_gray.c
    ${CMAKE_CURRENT_LIST_DIR}/to_int8/to_int8.c
    ${CMAKE_CURRENT_LIST_DIR}/common/memory_op.c)

set(PLATFORM_C_SRC
    ${CMAKE_CURRENT_LIST_DIR}/resize_image/resize_image_uc_implement.c)

#find_package(OpenMP)
#if(NOT TARGET OpenMP::OpenMP_CXX)
#    message(STATUS "cannot find OpenMP, check your cmake version at first, it should be > 3.9")
#else()
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    if (MSVC)
#        set(OpenMP_C_FLAGS "/openmp ${OpenMP_C_FLAGS}")
#        set(OpenMP_CXX_FLAGS "/openmp ${OpenMP_CXX_FLAGS}")
#    endif()
#endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x86)|(x64)|(X64)|(amd64)|(AMD64)|(i686)|(i386)")
    if (APPLE)
        add_definitions(-fno-stack-check) # Workaround a Xcode 11.X bug
    endif()
    message(STATUS "${CMAKE_SYSTEM_PROCESSOR}: Open SSE")
    FILE(GLOB DLCV_PROC_X8664_SRC ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/x86_64/*.c)
    set(DLCV_PROC_X8664_SRC 
        ${PLATFORM_C_SRC}
        ${DLCV_PROC_X8664_SRC})
    FILE(GLOB DLCV_PROC_AVX_SRC   ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/x86_64/avx/*.c)
    FILE(GLOB DLCV_PROC_SSE_SRC   ${CMAKE_CURRENT_LIST_DIR}/processor/cpu/x86_64/sse/*.c)
    add_library(DLCVPROCX8664 OBJECT ${DLCV_PROC_X8664_SRC})
    add_library(DLCVPROCAVX   OBJECT ${DLCV_PROC_AVX_SRC})
    add_library(DLCVPROCSSE   OBJECT ${DLCV_PROC_SSE_SRC})
    add_dependencies(DLCVPROCX8664 DLCVPROCAVX DLCVPROCSSE)
    if(MSVC)
        target_compile_options(DLCVPROCAVX PRIVATE /arch:AVX2)
    else()
        target_compile_options(DLCVPROCSSE   PRIVATE -msse4.1)
        target_compile_options(DLCVPROCAVX   PRIVATE -mavx2 -mfma)
        target_compile_options(DLCVPROCX8664 PRIVATE -msse4.1)
    endif()
    list(APPEND DLCV_PROC_OBJECTS_TO_LINK $<TARGET_OBJECTS:DLCVPROCX8664> $<TARGET_OBJECTS:DLCVPROCAVX> $<TARGET_OBJECTS:DLCVPROCSSE>)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    message(STATUS "Enabling AArch32 Assemblies")
    add_library(DLCV_PROC_ARM OBJECT ${PLATFORM_C_SRC} ${DLCV_PROC_ARM_C_SRC} ${DLCV_PROC_AArch32_SRC})
    list(APPEND DLCV_PROC_OBJECTS_TO_LINK $<TARGET_OBJECTS:DLCV_PROC_ARM>)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64")
    message(STATUS "Enabling AArch64 Assemblies")
    message(STATUS "DLCV_PROC_AArch64_SRC: ${DLCV_PROC_AArch64_SRC}")
    add_library(DLCV_PROC_ARM OBJECT ${PLATFORM_C_SRC} ${DLCV_PROC_ARM_C_SRC} ${DLCV_PROC_AArch64_SRC})
    list(APPEND DLCV_PROC_OBJECTS_TO_LINK $<TARGET_OBJECTS:DLCV_PROC_ARM>)
endif()

list(APPEND DLCV_PROC_OPT_OBJ_TO_LINK ${DLCV_PROC_OBJECTS_TO_LINK})
add_library(${DLCV_PROC_OPT} OBJECT ${DLCV_PROC_OPT_SRC} ${DLCV_PROC_OBJECTS_TO_LINK})
#add_library(${DLCV_PROC_OPT} STATIC ${DLCV_PROC_OPT_SRC} ${DLCV_PROC_OBJECTS_TO_LINK})
list(APPEND DLCV_PROC_OPT_OBJ_TO_LINK $<TARGET_OBJECTS:${DLCV_PROC_OPT}>)

#if(CMAKE_SYSTEM_NAME MATCHES "^Android")
#  target_link_libraries(${DLCV_PORC_OPT} PUBLIC android m)
#endif()

#enable_testing()
#add_subdirectory(test)