cmake_minimum_required(VERSION 3.5)

if(ANDROID)
  LIST(APPEND OpenCV_LIBRARIES  ${OpenCV_LIB_DIRS}/libopencv_highgui.so   ${OpenCV_LIB_DIRS}/libopencv_imgproc.so     ${OpenCV_LIB_DIRS}/libopencv_core.so     ${OpenCV_LIB_DIRS}/libopencv_imgcodecs.so ${OpenCV_LIB_DIRS}/libopencv_videoio.so)
  string_status("OpenCV_LIBRARIES" ${OpenCV_LIBRARIES})
  set(ANDROID_LOG log)
endif()

add_executable(detect_batch string_func.cpp detect_batch.cpp)
target_link_libraries(detect_batch ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(keypoints_batch string_func.cpp keypoints_batch.cpp)
target_link_libraries(keypoints_batch ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(multiclass_batch string_func.cpp multiclass_batch.cpp)
target_link_libraries(multiclass_batch ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(reidfeature_batch string_func.cpp reidfeature_batch.cpp)
target_link_libraries(reidfeature_batch ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

set (PREPROCESS_SRC
     preprocess_image.cpp
     ../src/common/util.cpp
     ../src/common/logging.cpp
     ../src/preprocess/preprocess.cpp)
add_executable(preprocess_image ${PREPROCESS_SRC})
target_link_libraries(preprocess_image ${OpenCV_LIBRARIES} ${ANDROID_LOG} ${DLCV_PROC_OPT_OBJ_TO_LINK})
add_dependencies(preprocess_image ${DLCV_PROC_OPT})

install (TARGETS preprocess_image DESTINATION tools)
install (TARGETS detect_batch DESTINATION tools)
install (TARGETS multiclass_batch DESTINATION tools)
install (TARGETS reidfeature_batch DESTINATION tools)
