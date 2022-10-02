
#include <iostream>
#include "opencv2/opencv.hpp"

#include <gtest/gtest.h>
#include "modelrunner.h"
#include "dlcv_testcase.h"
#include "test_common.h"
#include "filesystem/path.h"
#include "filesystem/resolver.h"

using namespace std;
using namespace vision;

#define  MODEL_ZOO_ROOT  "model_zoo"

void run_session(std::string& cfg_file, std::string& result_file) {

    std::shared_ptr<ModelRunner> det(new ModelRunner(cfg_file));
    
    DlcvTestCase testcase(result_file.c_str());
    for (auto item = testcase.case_list.begin(); item != testcase.case_list.end(); item++) {
        std::string image_path = item->first;
        vision::DLCVOut &historyOut= item->second;

        filesystem::path cfg_file_path(cfg_file);
        image_path = "/" + image_path;
        image_path = cfg_file_path.parent_path().str() + image_path;
        vision::DLCVOut currentOut;
        cv::Mat im = cv::imread(image_path);
        EXPECT_GT(im.rows, 0);


        det->run(im, currentOut);
        testcase.CheckOutput(currentOut, historyOut);
    }
}

TEST(tensorrt, modelzoo) {
    std::vector<std::string> folderList;
    const char* detect_folder = MODEL_ZOO_ROOT;
    RecursiveFolder(detect_folder, folderList);
    for (auto& folder : folderList) {
        std::vector<std::string> suffixList = { "json" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        for (auto& f : fileList) {
            int find_res = f.find("dlcv.json");
            if (find_res < 0)
                continue;
            printf("%s\n", f.c_str());

            std::string cfg_file = f;
            filesystem::path cfg_file_path(cfg_file);
            std::string result_file = (cfg_file_path.parent_path() / filesystem::path("unit_test_result.json")).str();

            run_session(cfg_file, result_file);
        }
    }
}

//
//TEST(tensorrt, dlcv) {
//    std::vector<std::string> folderList;
//    const char* detect_folder = TENSORRT_MODEL_ROOT;
//    RecursiveFolder(detect_folder, folderList);
//    for (auto& folder : folderList) {
//        std::vector<std::string> suffixList = { "json" };
//        std::vector<std::string> fileList;
//        RecursiveFile(folder.c_str(), suffixList, fileList);
//        for (auto& f : fileList) {
//            int find_res = f.find("dlcv.json");
//            if (find_res < 0)
//                continue;
//
//            find_res = f.find("face_kps_v1");
//            if (find_res < 0)
//                continue;
//
//            printf("%s\n", f.c_str());
//            
//            std::string cfg_file = f;
//            filesystem::path cfg_file_path(cfg_file);
//            std::string cfg_file_name = cfg_file_path.filename();
//            std::string cfg_file_dir = cfg_file_path.parent_path().str();
//            std::string result_file = cfg_file_dir + "/unit_test_result.json";
//
//            run_session(cfg_file, result_file);
//        }
//    }
//}
//
//void livecheck_preprocess(cv::Mat &mat, cv::Rect rect, cv::Mat &output_mat, int inW, int inH) {
//    int x = (std::max)(rect.x, 0);
//    int y = (std::max)(rect.y, 0);
//    int w = (std::min)(rect.width, mat.cols);
//    int h = (std::min)(rect.height, mat.rows);
//
//    cv::Rect rect1(x, y, w, h);
//    cv::Mat mat1 = mat(rect1);
//
//    cv::Mat mat_border;
//    cv::copyMakeBorder(mat, mat_border, h, h, w, w, cv::BORDER_REFLECT);
//
//    cv::Rect rect2(x, y, 3 * w, 3 * h);
//    cv::Mat mat2 = mat_border(rect2);
//
//    cv::Mat mat1_resize;
//    cv::resize(mat1, mat1_resize, cv::Size(inW, inH));
//
//    cv::Mat mat2_resize;
//    cv::resize(mat2, mat2_resize, cv::Size(inW, inH));
//
//    cv::Mat mat1_8u, mat2_8u;
//    mat1_resize.convertTo(mat1_8u, CV_32FC3);
//    mat2_resize.convertTo(mat2_8u, CV_32FC3);
//
//    // 6*224*224
//    float* img_data = new float[3 * inW * inH];
//    float* img_pad_data = new float[3 * inW*inH];
//
//    cv::Mat B(inH, inW, CV_32FC1, img_data);
//    cv::Mat G(inH, inW, CV_32FC1, img_data + inW * inH);
//    cv::Mat R(inH, inW, CV_32FC1, img_data + inW * inH * 2);
//    std::vector<cv::Mat> data1{ B, G, R };
//    cv::split(mat1_8u, data1);
//
//    cv::Mat B1(inH, inW, CV_32FC1, img_pad_data);
//    cv::Mat G1(inH, inW, CV_32FC1, img_pad_data + inW * inH);
//    cv::Mat R1(inH, inW, CV_32FC1, img_pad_data + inW * inH * 2);
//    std::vector<cv::Mat> data2{ B1, G1, R1 };
//    cv::split(mat2_8u, data2);
//
//    float mean[] = { 128.0f, 128.0f, 128.0f };
//    cv::Scalar_<float> m_bn_mean = cvScalar(mean[0], mean[1], mean[2]);
//    for (int ix = 0; ix < 3; ++ix) {
//        data1[ix] = (data1[ix] - m_bn_mean[ix]);
//        data2[ix] = (data2[ix] - m_bn_mean[ix]);
//    }
//
//    cv::Mat hconcat_mat(6, 224, CV_32FC(224));
//    memcpy(hconcat_mat.data, img_data, 3 * inW*inH * sizeof(float));
//    memcpy(hconcat_mat.data + 3 * inW*inH*sizeof(float), img_pad_data, 3 * inW*inH * sizeof(float));
//    output_mat = hconcat_mat;
//
//    delete[]img_data;
//    delete[]img_pad_data;
//}
//
//TEST(tensorrt, custom_preprocess) {
//
//    std::string dlcv_json_livecheck = TENSORRT_MODEL_ROOT"/live_demo_face_kp/dlcv_custom.json";
//
//    std::string cfg_file = dlcv_json_livecheck;
//    filesystem::path cfg_file_path(cfg_file);
//    std::string cfg_file_name = cfg_file_path.filename();
//    std::string cfg_file_dir = cfg_file_path.parent_path().str();
//    std::string result_file = cfg_file_dir + "/unit_test_result.json";
//
//    std::shared_ptr<ModelRunner> livecheck(new ModelRunner(dlcv_json_livecheck));
//    
//    DlcvTestCase testcase(result_file.c_str());
//    for (auto item = testcase.case_list.begin(); item != testcase.case_list.end(); item++) {
//        std::string image_path = item->first;
//        std::string image_dir = dlcv_json_livecheck;
//
//        filesystem::path img_file_path(image_dir);
//        image_dir = img_file_path.parent_path().str();
//
//        vision::DLCVOut &historyOut = item->second;
//        image_path = "/" + image_path;
//        image_path = image_dir + image_path;
//
//        cv::Mat im = cv::imread(image_path);
//        EXPECT_GT(im.rows, 0);
//
//        cv::Rect new_rect = {320,842,605,605};
//
//        // custom_preprocess
//        {
//            cv::Mat newMat;
//            livecheck_preprocess(im, new_rect, newMat, 224, 224);
//            vision::DLCVOut classOutput;
//            livecheck->run(newMat,classOutput);
//            testcase.CheckOutput(classOutput, historyOut);
//        }
//        
//    }
//}
//
//
//TEST(tensorrt, custom_postprocess) {
//
//
//    std::string dlcv_json_livecheck = TENSORRT_MODEL_ROOT"/fd_RFB_960x960_opset9/dlcv_custom.json";
//
//    std::string cfg_file = dlcv_json_livecheck;
//    filesystem::path cfg_file_path(cfg_file);
//    std::string cfg_file_name = cfg_file_path.filename();
//    std::string cfg_file_dir = cfg_file_path.parent_path().str();
//
//    std::string result_file = cfg_file_dir + "/unit_test_result.json";
//
//    std::shared_ptr<ModelRunner> multi_face_detect(new ModelRunner(dlcv_json_livecheck));
//
//    DlcvTestCase testcase(result_file.c_str());
//    for (auto item = testcase.case_list.begin(); item != testcase.case_list.end(); item++) {
//        std::string image_path = item->first;
//        std::string image_dir = dlcv_json_livecheck;
//
//        filesystem::path img_file_path(image_dir);
//        image_dir = img_file_path.parent_path().str();
//
//        vision::DLCVOut &historyOut = item->second;
//        image_path = "/" + image_path;
//        image_path = image_dir + image_path;
//
//        cv::Mat im = cv::imread(image_path);
//        EXPECT_GT(im.rows, 0);
//
//        // custom_preprocess
//        {
//            vision::DLCVOut classOutput;
//            multi_face_detect->run(im, classOutput);
//            testcase.CheckOutput(classOutput, historyOut);
//        }
//
//    }
//}