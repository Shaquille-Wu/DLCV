
#include <iostream>
#include <libgen.h>
#include "detector.h"
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include "logging.h"
#include <gtest/gtest.h>
#include "util.h"
#include "test_common.h"
#include "blob.h"               // for Blob
#include "dlcv.h"               // for ElementType, F32
#include "json.hpp"               // for ElementType, F32



using namespace std;
using namespace vision;

#define  MODELZOO_ROOT  "./model_zoo/"
#define  DETECT_MODEL_ROOT  "./model_zoo/detection"

using json=nlohmann::json;

struct DetectResult {
    std::string type;
    float   score;
    float   x1;
    float   y1;
    float   x2;
    float   y2;
};

static json load_json(const string config_file){
    json cfg;
    try {
        std::ifstream inJsonFile(config_file.c_str());
        if (!inJsonFile) {
            LOG(ERROR) << " Error: json file open failed: " << config_file;
            ABORT();
        }
        inJsonFile >> cfg;
    }
    catch (std::exception& e) {
        LOG(ERROR) << " Error: json file read failed: " << e.what();
        ABORT();
    }
    return cfg;
}

static void parse_result_json(std::string& result_file, cv::Mat& test_mat, std::vector<DetectResult>& detect_result) {
    std::string result_folder = get_file_path(result_file.c_str());
    std::cout << result_file << std::endl;
    json cfg = load_json(result_file);

    std::string input_image = cfg.at("input");
    std::string image = result_folder + input_image;
    test_mat = cv::imread(image);
    std::cout << image << std::endl;
    std::cout <<"test_mat cols:" <<  test_mat.cols << std::endl;

    vector<json> results = cfg.at("result");
    detect_result.resize(results.size());
    for(int i=0;i<results.size();++i) {
        detect_result[i].type = results[i].at("type");
        detect_result[i].score = results[i].at("score");
        detect_result[i].x1 = results[i].at("x1");
        detect_result[i].y1 = results[i].at("y1");
        detect_result[i].x2 = results[i].at("x2");
        detect_result[i].y2 = results[i].at("y2");
    }
}

static void check_detect_result(std::string& cfg_file, std::string& result_file) {
    cv::Mat im; 
    std::vector<DetectResult> detect_result;
    parse_result_json(result_file, im, detect_result);

    Detector* det = new Detector(cfg_file);
    std::map<std::string, std::vector<Box> > boxes_map;
    int r = det->run(im, boxes_map);

    int boxes_num = 0;
    for(auto it = boxes_map.begin(); it != boxes_map.end(); ++it) {
        boxes_num += it->second.size();
    }
    ASSERT_EQ(boxes_num, detect_result.size());

    int idx = 0;
    for(auto it = boxes_map.begin(); it != boxes_map.end(); ++it) {
        std::vector<Box>& boxes = (it->second);
        for(auto& b:boxes) {
            DetectResult& result = detect_result[idx++];
            //std::cout << it->first << " " << b.score << " " << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2 << std::endl;
            EXPECT_EQ(result.type, it->first);
            EXPECT_NEAR(result.score, b.score, 0.001);
            EXPECT_NEAR(result.x1, b.x1, 0.001);
            EXPECT_NEAR(result.y1, b.y1, 0.001);
            EXPECT_NEAR(result.x2, b.x2, 0.001);
            EXPECT_NEAR(result.y2, b.y2, 0.001);
        }
    }
}

TEST(detector, ncnn) {
#if 0
    struct  timeval   tv_start      = { 0, 0 } ;
    struct  timeval   tv_end        = { 0, 0 } ;
    long long int     detect_time   = 0;

    std::string cfgfile = MODELZOO_ROOT"detection/face_ssdlite1_qf_0.35_r2.0/ncnn/det_face_ssdlite1_qf_r2.0.ncnn.json";
    std::string imgfile = MODELZOO_ROOT"detection/face_ssdlite1_qf_0.35_r2.0/ssd_fp_debug.png";
    std::string label_name = "face";
    int run_loop = 1;

    LOG(INFO) << "config file: " << cfgfile;
    LOG(INFO) << "image file:  " << imgfile;
    LOG(INFO) << "label name:  " << label_name;
    LOG(INFO) << "run loop:    " << run_loop;

    Ddlcvetector* det = new Detector(cfgfile);
    cv::Mat im = cv::imread(imgfile);
    std::map<std::string, std::vector<Box> > boxes_map;
    gettimeofday(&tv_start,0);
    for(int i = 0 ; i < run_loop ; i ++)
    {
        boxes_map.clear();
        int r = det->run(im, boxes_map);
        EXPECT_EQ(r,0);
    }
    //gettimeofday(&tv_end,0);
    //detect_time = 1000000 * (tv_end.tv_sec-tv_start.tv_sec)+ tv_end.tv_usec-tv_start.tv_usec;
    //LOG(INFO) << "detect time eplased: " << detect_time;

    std::map<std::string, std::vector<Box> >::iterator    iter = boxes_map.find(label_name);
    EXPECT_TRUE(boxes_map.end() != iter);
    
    char output_log[256] = { 0 };
    const std::vector<Box>& boxes = (iter->second);
    EXPECT_TRUE(boxes.size() > 0);

    for (unsigned int i = 0; i < boxes.size(); i++) {
        Box b = boxes[i];
        //snprintf(output_log, 256, "%03d %d %.6f %.6f %.6f %.6f %.6f", i, b.cls, b.score, b.x1, b.y1, b.x2, b.y2);
        //LOG(INFO) << output_log;

        if (i == 0){
            EXPECT_EQ(1, b.cls);
            EXPECT_NEAR(0.99, b.score, 0.1);
            EXPECT_NEAR(317, b.x1, 1);
            EXPECT_NEAR(113, b.y1, 1);
            EXPECT_NEAR(580, b.x2, 1);
            EXPECT_NEAR(346, b.y2, 1);
        }
    }
#endif
}

TEST(detector, snpe_1_25) {
    std::string snpe_folder = "snpe-1.25";
    std::vector<std::string> folderList;
    const char* detect_folder = DETECT_MODEL_ROOT;
    RecursiveFolder(detect_folder, snpe_folder, folderList);
    for(auto& folder:folderList) {
        std::vector<std::string> suffixList = { "json" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        for(auto& f:fileList) {
            int find_res = f.find(".snpe");
            if(find_res < 0)
                continue;
            std::string cfg_file            = f;
            std::string cfg_file_dir        = "";
            std::string result_file         = "unit_test_result";
            int         last_slope_dir_pos  = cfg_file.rfind("/"); 
            std::string cfg_file_name       = "";
            if(last_slope_dir_pos >= 0)
            {
                cfg_file_dir   = cfg_file.substr(0, last_slope_dir_pos + 1);
                cfg_file_name  = cfg_file.substr(last_slope_dir_pos + 1, cfg_file.length() - last_slope_dir_pos);
            }
            else
            {
                cfg_file_name  = cfg_file;
            }

            if(((int)(cfg_file_name.find("_gpu16.snpe"))) >= 0)
                result_file += "_gpu16";
            else if(((int)(cfg_file_name.find("_gpu.snpe"))) >= 0)
                result_file += "_gpu";
            else if(((int)(cfg_file_name.find("_quantized.snpe"))) >= 0)
                result_file += "_quantized";
            else
                continue;
            result_file   = cfg_file_dir + result_file + ".json";
            check_detect_result(cfg_file, result_file);
        }
    }  
}

TEST(detector, snpe_1_29) {
    std::string snpe_folder = "snpe-1.29";
    std::vector<std::string> folderList;
    const char* detect_folder = DETECT_MODEL_ROOT;
    RecursiveFolder(detect_folder, snpe_folder, folderList);
    for(auto& folder:folderList) {
        std::vector<std::string> suffixList = { "json" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        for(auto& f:fileList) {
            int find_res = f.find(".snpe");
            if(find_res < 0)
                continue;
            std::string cfg_file            = f;
            std::string cfg_file_dir        = "";
            std::string result_file         = "unit_test_result";
            int         last_slope_dir_pos  = cfg_file.rfind("/"); 
            std::string cfg_file_name       = "";
            if(last_slope_dir_pos >= 0)
            {
                cfg_file_dir   = cfg_file.substr(0, last_slope_dir_pos + 1);
                cfg_file_name  = cfg_file.substr(last_slope_dir_pos + 1, cfg_file.length() - last_slope_dir_pos);
            }
            else
            {
                cfg_file_name  = cfg_file;
            }

            if(((int)(cfg_file_name.find("_gpu16.snpe"))) >= 0)
                result_file += "_gpu16";
            else if(((int)(cfg_file_name.find("_gpu.snpe"))) >= 0)
                result_file += "_gpu";
            else if(((int)(cfg_file_name.find("_quantized.snpe"))) >= 0)
                result_file += "_quantized";
            else
                continue;
            result_file   = cfg_file_dir + result_file + ".json";
            check_detect_result(cfg_file, result_file);
        }
    }  
}

TEST(detector, snpe_1_36) {
    std::string snpe_folder = "snpe-1.36";
    std::vector<std::string> folderList;
    const char* detect_folder = DETECT_MODEL_ROOT;
    RecursiveFolder(detect_folder, snpe_folder, folderList);
    for(auto& folder:folderList) {
        std::vector<std::string> suffixList = { "json" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        for(auto& f:fileList) {
            int find_res = f.find(".snpe");
            if(find_res < 0)
                continue;
            std::string cfg_file            = f;
            std::string cfg_file_dir        = "";
            std::string result_file         = "unit_test_result";
            int         last_slope_dir_pos  = cfg_file.rfind("/"); 
            std::string cfg_file_name       = "";
            if(last_slope_dir_pos >= 0)
            {
                cfg_file_dir   = cfg_file.substr(0, last_slope_dir_pos + 1);
                cfg_file_name  = cfg_file.substr(last_slope_dir_pos + 1, cfg_file.length() - last_slope_dir_pos);
            }
            else
            {
                cfg_file_name  = cfg_file;
            }
            if(((int)(cfg_file_name.find("_gpu16.snpe"))) >= 0)
                result_file += "_gpu16";
            else if(((int)(cfg_file_name.find("_gpu.snpe"))) >= 0)
                result_file += "_gpu";
            else if(((int)(cfg_file_name.find("_quantized.snpe"))) >= 0)
                result_file += "_quantized";
            else
                continue;
            result_file   = cfg_file_dir + result_file + ".json";
            check_detect_result(cfg_file, result_file);
        }
    }  
}

TEST(detector, openvino) {
    std::string openvino_folder = "openvino_2020.2.120";
    std::vector<std::string> folderList;
    const char* detect_folder = DETECT_MODEL_ROOT;
    RecursiveFolder(detect_folder, openvino_folder, folderList);
    for(auto& folder:folderList) {
        std::vector<std::string> suffixList = { "xml" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        EXPECT_EQ(1, fileList.size());
        for(auto& f:fileList) {
            std::string cfg_file = f.substr(0, f.size() - 4) + ".json";
            std::string result_file = folder + "/unit_test_result.json";
            check_detect_result(cfg_file, result_file);
        }
    }
}

TEST(detector, mnn) {
    std::string mnn_folder = "mnn";
    std::vector<std::string> folderList;
    const char* detect_folder = DETECT_MODEL_ROOT;
    RecursiveFolder(detect_folder, mnn_folder, folderList);
    for(auto& folder:folderList) {
        std::vector<std::string> suffixList = { "json" };
        std::vector<std::string> fileList;
        RecursiveFile(folder.c_str(), suffixList, fileList);
        for(auto& f:fileList) {
            int find_res = f.find(".mnn");
            if(find_res < 0)
                continue;
            std::string cfg_file            = f;
            std::string cfg_file_dir        = "";
            std::string result_file         = "unit_test_result";
            int         last_slope_dir_pos  = cfg_file.rfind("/"); 
            std::string cfg_file_name       = "";
            if(last_slope_dir_pos >= 0)
            {
                cfg_file_dir   = cfg_file.substr(0, last_slope_dir_pos + 1);
                cfg_file_name  = cfg_file.substr(last_slope_dir_pos + 1, cfg_file.length() - last_slope_dir_pos);
            }
            else
            {
                cfg_file_name  = cfg_file;
            }
            if(((int)(cfg_file_name.find("_cpu_fp32.mnn"))) >= 0)
                result_file += "_cpu_fp32";
            else if(((int)(cfg_file_name.find("_cpu_fp16.mnn"))) >= 0)
                result_file += "_cpu_fp16";
            else if(((int)(cfg_file_name.find("_gpu_fp32.mnn"))) >= 0)
                result_file += "_gpu_fp32";
            else if(((int)(cfg_file_name.find("_gpu_fp16.mnn"))) >= 0)
                result_file += "_gpu_fp16";
            else
                continue;
            result_file   = cfg_file_dir + result_file + ".json";
            check_detect_result(cfg_file, result_file);
        }
    }
}