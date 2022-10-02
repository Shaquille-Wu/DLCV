/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file keypoints_batch.cpp
 * @brief a tools which can test batch-images for detect
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-05-28
 */

#include <iostream>
#include "keypoints_attributes.h"
#include "opencv2/opencv.hpp"
#include <getopt.h>
#include <sys/time.h>
#include "string_func.h"
#include "logging.h"

using namespace std;
using namespace vision;

static void show_result(int                                       image_idx,
                        const char*                               image_file_name,
                        std::vector<cv::Point> const&             kpts_data,
                        std::vector<std::vector<float> > const&   attr_data,
                        char*                                     log_buf,
                        int&                                      log_buf_len,
                        long long int                             forward_cost)
{
    int  i                = 0;
    int  j                = 0;
    char cur_log[8192]    = { 0 };
    int  cur_log_len      = 0;
    int  point_count      = (int)(kpts_data.size());
    int  attri_count      = (int)(attr_data.size());
    
    LOG(INFO) << "Keypoints result:";
    std::string outdata = "\n";

    cur_log_len += sprintf(cur_log, "%s, ", image_file_name);
    for(i = 0 ; i < point_count ; i ++)
        cur_log_len += sprintf(cur_log + cur_log_len, "%d, %d, ", kpts_data[i].x, kpts_data[i].y);
    for(i = 0 ; i < attri_count ; i ++)
    {
        int cur_attr_cnt = (int)(attr_data[i].size());
        for(j = 0 ; j < cur_attr_cnt ; j ++)
            cur_log_len += sprintf(cur_log + cur_log_len, "%.7f, ", attr_data[i][j]);
    }
    cur_log[cur_log_len - 2] = 0;
    cur_log[cur_log_len - 1] = 0;
    cur_log_len              = cur_log_len - 2;
    log_buf_len             += sprintf(log_buf + log_buf_len, "%s\n", cur_log);
    printf("%05d, %s\n", image_idx, cur_log);
}

static int forward_image(vision::KeypointsAttributes*    kpts_attr,
                         int                             image_idx, 
                         const char*                     image_file_dir,
                         const char*                     image_file_name, 
                         char*                           log_buf,
                         long long int&                  forward_cost)
{   
    int                 res                   = 0;
    int                 i                     = 0;
    std::string         image_file_path_name  = std::string(image_file_dir) + std::string("/") + std::string(image_file_name);
    cv::Mat             raw_image             = cv::imread(image_file_path_name);
    struct  timeval     tv_start              = { 0, 0 } ;
    struct  timeval     tv_end                = { 0, 0 } ;
    int                 log_buf_len           = 0;
    std::map<std::string, std::vector<Box> > boxes_map;
    if(nullptr == raw_image.data)
    {
        printf("cannot read image data\r\n");
        return -1;
    }

    std::vector<cv::Point>            keypoints;
    std::vector<std::vector<float> >  attributes;
    gettimeofday(&tv_start,0);
    res = kpts_attr->run(raw_image, keypoints, attributes);
    if(0 != res)
    {
        printf("forward failed\n");
        return -1;
    }
    gettimeofday(&tv_end,0);
    forward_cost = 1000000 * (tv_end.tv_sec-tv_start.tv_sec)+ tv_end.tv_usec-tv_start.tv_usec;
    printf("forward cost: %lld\r\n", forward_cost);
    show_result(image_idx,
                image_file_name,
                keypoints, 
                attributes,
                log_buf,
                log_buf_len, 
                forward_cost);

    return log_buf_len;
}

int main(int argc, char* argv[]){
    std::string   cfg_file;
    std::string   img_file_dir;
    int           run_loop       = 0;
    int           opt            = 0;
    while ((opt = getopt(argc, argv, "hj:i:n:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
               LOG(INFO) 
                        << "\nDESCRIPTION:\n"
                        << "  -j  confige file(.json file).\n"
                        << "  -i  image directory.\n"
                        << "  -n  run image count.\n"
                        << "\n";
                std::exit(0);
                break;
            case 'j':
                cfg_file       = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;                
            case 'i':
                img_file_dir   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'n':
                run_loop       = atoi(trim_string((nullptr == optarg ? std::string("") : std::string(optarg))).c_str());
                break;                 
            default:
                LOG(INFO) << "Invalid parameter specified.";
                std::exit(-1);
        }
    }     

    if(run_loop <= 0)
        run_loop = 0;

    std::cout << "cfgfile: " << cfg_file << std::endl;
    std::cout << "imgdir:  " << img_file_dir << std::endl;
    std::cout << "imgcnt:  " << run_loop << std::endl;

    int                         i                     = 0;
    long long int               cur_forward_cost      = 0;
    long long int               total_forward_cost    = 0;
    static const int            LOG_BUF_SIZE          = (64 << 20);
    int                         log_offset            = 0;
    char*                       log_buf               = new char[LOG_BUF_SIZE];
    FILE*                       log_file              = 0;
    std::string                 log_file_name;
    long long int               forward_cost_min      = 0x7FFFFFFFFFFFFFFF;
    long long int               forward_cost_max      = 0;

    auto                              kps_attr = new vision::KeypointsAttributes(cfg_file);
    std::vector<std::string>          image_list;

    image_list = travel_image_dir(img_file_dir, 
                                  std::vector<std::string>({std::string(".jpg"), std::string(".jpeg"), std::string(".bmp"), std::string(".png"), std::string(".pgm"), std::string(".ppm")}), 
                                  std::string(""));
    LOG(INFO) << "there " << (int)(image_list.size()) << " files in " << img_file_dir;
    if(0 == image_list.size())
        goto DLCV_KPTS_BATCH_LEAVE;
    memset(log_buf,        0, LOG_BUF_SIZE);

    if(0 == run_loop)
        run_loop = (int)(image_list.size());

    for(i = 0 ; i < run_loop; i ++)
    {
        cur_forward_cost  = 0;
        int cur_res       = forward_image(kps_attr,
                                          i, 
                                          img_file_dir.c_str(),
                                          image_list[i].c_str(),
                                          log_buf + log_offset,
                                          cur_forward_cost);
        if(cur_res <= 0)
        {
            LOG(INFO) << i << ", " << image_list[i] << ", error";
        }
        else
        {
            if(forward_cost_min > cur_forward_cost)
                forward_cost_min = cur_forward_cost;
            if(forward_cost_max < cur_forward_cost)
                forward_cost_max = cur_forward_cost;
            log_offset         += cur_res;
        }
        total_forward_cost += cur_forward_cost;
    }
    LOG(INFO) << "total image count: " << i << ", average cost " << (total_forward_cost / run_loop);

    if(log_offset > 0)
    {
        log_file_name = "./batch_result.txt";
        log_file = fopen(log_file_name.c_str(), "wb");
        fwrite(log_buf, log_offset, 1, log_file);
        fflush(log_file);
        fclose(log_file);
    }


DLCV_KPTS_BATCH_LEAVE:
    if(nullptr != log_buf)
        delete[] log_buf;

    return 0;
}
