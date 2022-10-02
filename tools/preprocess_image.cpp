/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file preprocess_image.cpp
 * @brief preprocess image tool.
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-05-21
 */

#include <iostream>
#include <getopt.h>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

#include "logging.h"
#include "blob.h"               // for Blob
#include "dlcv.h"               // for DLCVOut, Segment, ElementType, U8
#include "json.hpp"             // for operator""_json
#include "preprocess.h"         // for PreprocInfo


static const int     FAILURE    = 1;
static const int     SUCCESS    = 0;

static std::string trim_string(const std::string& src)
{
    int    str_len     = src.length();
    int    start_pos   = 0;
    int    end_pos     = 0;
    int    i           = 0;
    char*  trim_str    = new char[str_len + 1];
    memset(trim_str, 0, str_len + 1);
    for(i = 0 ; i < str_len ; i ++)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i] && '\t' != (src.c_str())[i])
        {
            start_pos = i;
            break;
        }
    }
    for(i = str_len - 1 ; i >= 0 ; i --)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i] && '\t' != (src.c_str())[i])
        {
            end_pos = i;
            break;
        }
    }

    int trim_len = end_pos - start_pos + 1;
    for(i = 0 ; i < trim_len ; i ++)
        trim_str[i] = (src.c_str())[i + start_pos];

    std::string  result = std::string(trim_str);
    delete[] trim_str;

    return result;
}

static std::string get_file_base_name(const std::string& file_name) noexcept
{
    std::string  file_base_name;
    int          pos        = file_name.rfind(".");
    if(pos < 0)
        return "";
    
    file_base_name = file_name.substr(0, pos);

    return file_base_name;
}

static std::string               get_file_ext(const std::string& src)
{
    int    len        = src.length();
    char*  ext        = new char[len + 1];
    int    start_pos  = 0;
    int    i          = 0;

    for(i = len - 1 ; i >= 0 ; i --)
    {
        if('.' == src.c_str()[i])
        {
            start_pos = i;
            break;
        }
    }

    memset(ext, 0, len + 1);
    for(i = start_pos ; i < len ; i ++)
    {
        ext[i - start_pos] = src.c_str()[i];
    }

    std::string   result = ext;

    delete[] ext;
    return result;
}

static nlohmann::json load_json(const std::string config_file){
    nlohmann::json cfg;
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

static int run_image(vision::Preprocessor*       preprocessor, 
                     std::vector<vision::Blob>&  raw_data,
                     const std::string&          src_image_name,  
                     const std::string&          dst_image_name)
{
    cv::Mat       src_img         = cv::imread(src_image_name.c_str());
    if(true == src_img.empty())
    {
        LOG(ERROR) << "image file is invalid " << src_image_name;
        return -1;
    }

    vision::PreprocInfo pre_proc_info;
    preprocessor->run(src_img, pre_proc_info, raw_data[0]);

    float*  data_ptr      = raw_data[0].get<float>();
    int     raw_data_size = raw_data[0].num_elements() * raw_data[0].element_size();
    FILE*   raw_file = fopen(dst_image_name.c_str(), "wb");
    if(0 == raw_file)
    {
        LOG(ERROR) << "dst file name is invalid " << dst_image_name;
    }
    else
    {
        fwrite(data_ptr, raw_data_size, 1, raw_file);
        fflush(raw_file);
        fclose(raw_file);
    }
    
    return 0;
}

int main(int argc, char** argv)
{
    std::string   json_file            = "";
    std::string   src_image_name       = "";
    std::string   dst_image_name       = "";
    std::string   input_idx_str        = "";
    int           input_idx            = 0;
    bool          verbose_flag         = false;

    int opt = 0;
    while ((opt = getopt(argc, argv, "hj:s:d:v:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "  -j  json file, identical to engine's json file\n"
                        << "  -s  source image\n"
                        << "  -d  destinate image, system will make it the same as source image, if it is empty, but its ext-name will be \".raw\"\n"
                        << "  -i  the index of input, if there are multiply inputs in the json"
                        << "  -v  verbos"
                        << "\n"
                        << std::endl;
                std::exit(SUCCESS);
                break;
            case 'j':
                json_file        = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 's':
                src_image_name   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;                
            case 'd':
                dst_image_name   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;    
            case 'i':
                input_idx_str    = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'v':
                verbose_flag     = true;
                break;                                            
            default:
                LOG(ERROR) << "Invalid parameter specified.";
                std::exit(FAILURE);
        }
    }    


    if("" == json_file)
    {
        LOG(ERROR) << "json_file cannot be empty";
        std::exit(FAILURE);
    }

    if("" == src_image_name)
    {
        LOG(ERROR) << "src_image_name cannot be empty";
        std::exit(FAILURE);
    }

    if("" == dst_image_name)
    {
        dst_image_name = get_file_base_name(src_image_name);
        dst_image_name = dst_image_name + ".raw";
    }  
    else
    {
        std::string ext_name = get_file_ext(dst_image_name);
        if(".raw" != trim_string(ext_name))
        {
            LOG(ERROR) << ext_name << " dst_image_name should be .raw";
            std::exit(FAILURE);
        }
    }

    if("" != input_idx_str)
        input_idx = atoi(input_idx_str.c_str());

    if(src_image_name == dst_image_name)
    {
        LOG(ERROR) << "src image is the same as dst image:";
        LOG(ERROR) << "src image: " << src_image_name;
        LOG(ERROR) << "dst image: " << dst_image_name;
        std::exit(FAILURE);
    }

    if(true == verbose_flag)
    {
        LOG(INFO) << "current input:";
        LOG(INFO) << "json file: " << json_file;
        LOG(INFO) << "src image: " << src_image_name;
        LOG(INFO) << "dst image: " << dst_image_name;
    }

    nlohmann::json                       cfg             = load_json(json_file);
    std::vector<vision::Preprocessor*>   preprocess(0);
    if(cfg.at("preprocess").is_array())
    {
        std::vector<nlohmann::json> json_vec       = cfg.at("preprocess");
        preprocess.resize(json_vec.size(), nullptr);
        for(int i = 0 ; i < (int)(json_vec.size()) ; i ++)
            preprocess[i]  = new vision::Preprocessor(json_vec[i]);
    }
    else
    {
        preprocess.resize(1, nullptr);
        preprocess[0] = new vision::Preprocessor(cfg.at("preprocess"));
    }
    nlohmann::json                       inference_json  = cfg.at("inference");
    std::vector<nlohmann::json>          incfg           = inference_json.at("inputs");
    const std::vector<int>               shape           = incfg[0].at("shape");
    std::vector<vision::Blob>            raw_data;
    CHECK(shape.size() == 4);

    if(input_idx >= ((int)(preprocess.size())))
    {
        input_idx = ((int)(preprocess.size())) - 1;
        LOG(INFO) << "input_idx is beyond the size of preprocess";
    }
        

    if (incfg[0].contains("dtype") && incfg[0].at("dtype") == "int8")
    {
        LOG(ERROR) << "output cannot be \"int8\"";
        return -1;
    } 
    else 
        raw_data.push_back(vision::Blob::create<float>(shape));


    int res = run_image(preprocess[input_idx], raw_data, src_image_name, dst_image_name);

    if(true == verbose_flag)
        LOG(INFO) << "end";

    for(int i = 0 ; i < ((int)(preprocess.size())) ; i ++)
        delete preprocess[i];
    preprocess.clear();

    return res;
}