cmake_minimum_required(VERSION 3.5)

if(ANDROID)
  LIST(APPEND OpenCV_LIBRARIES  ${OpenCV_LIB_DIRS}/libopencv_highgui.so   ${OpenCV_LIB_DIRS}/libopencv_imgproc.so     ${OpenCV_LIB_DIRS}/libopencv_core.so     ${OpenCV_LIB_DIRS}/libopencv_imgcodecs.so ${OpenCV_LIB_DIRS}/libopencv_videoio.so)
  string_status("OpenCV_LIBRARIES" ${OpenCV_LIBRARIES})
endif()

add_executable(test_detector test_detector.cpp)
target_link_libraries(test_detector ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_featuremap_runner test_featuremap_runner.cpp)
target_link_libraries(test_featuremap_runner ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_multiclass test_multiclass.cpp)
target_link_libraries(test_multiclass ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_reid test_reidfeature.cpp)
target_link_libraries(test_reid ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_keypoints_attribute test_keypoints_attributes.cpp)
target_link_libraries(test_keypoints_attribute ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_eco_featuremap test_eco_featuremap.cpp)
target_link_libraries(test_eco_featuremap ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_imodel_runner test_imodel_runner.cpp)
target_link_libraries(test_imodel_runner ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_dlcv_version test_dlcv_version.cpp)
target_link_libraries(test_dlcv_version ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_superpoint_postproc test_superpoint_postproc.cpp)
target_link_libraries(test_superpoint_postproc ${LIBDLCV} ${OpenCV_LIBRARIES} dl)

add_executable(test_stereomesh_runner test_stereomesh_runner.cpp)
target_link_libraries(test_stereomesh_runner ${LIBDLCV} ${OpenCV_LIBRARIES} dl)