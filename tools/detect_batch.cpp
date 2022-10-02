/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file detect_batch.cpp
 * @brief a tools which can test batch-images for detect
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-05-28
 */

#include<iostream>
#include "detector.h"
#include "opencv2/opencv.hpp"
#include <getopt.h>
#include <sys/time.h>
#include "string_func.h"
#include "logging.h"

using namespace std;
using namespace vision;

static void show_result(int                                              image_idx,
                        const char*                                      image_file_name,
                        std::map<std::string, std::vector<Box> >         result_data,
                        char*                                            log_buf,
                        int&                                             log_buf_len,
                        long long int                                    forward_cost)
{
    int  i                = 0;
    char cur_log[1024]    = { 0 };
    int  result_item_cnt  = 0;
    int  valid_count      = 0;
    
    std::vector<Box>                      vec_box;
    std::vector<std::string>              vec_label;
    std::vector<std::pair<float, int>>    vec_sort_idx;

    int    result_cnt = 0;
    int    j          = 0;
    std::map<std::string, std::vector<Box> >::iterator    iter = result_data.begin();
    while(result_data.end() != iter)
    {
        const std::vector<Box>& boxes      = (iter->second);
        result_cnt += ((int)(boxes.size()));
        iter ++;
    }
    vec_box.resize(result_cnt);
    vec_sort_idx.resize(result_cnt);
    vec_label.resize(result_cnt);
    iter = result_data.begin();
    while(result_data.end() != iter)
    {
        const std::vector<Box>& boxes      = (iter->second);
        for(i = 0 ; i < ((int)(boxes.size())) ; i ++)
        {
            vec_box[j]             = boxes[i] ;
            vec_label[j]           = (iter->first);
            vec_sort_idx[j].first  = vec_box[j].score;
            vec_sort_idx[j].second = j;
            j ++;
        }
        iter ++;
    }

    std::sort(vec_sort_idx.begin(), vec_sort_idx.end());
    result_item_cnt = j;

    if(result_item_cnt > 0)
    {
        for(i = 0 ; i < result_item_cnt ; i ++)
        {        
            int                 box_idx    = vec_sort_idx[result_item_cnt - i - 1].second;
            const Box&          b          = vec_box[box_idx];
            sprintf(cur_log, "%s, %s, %.7f, %.7f, %.7f, %.7f, %.7f\n", 
                            image_file_name, vec_label[box_idx].c_str(), b.score, b.x1, b.y1, b.x2, b.y2);
            log_buf_len += sprintf(log_buf + log_buf_len, "%s", cur_log);
            printf("%05d, %s", i, cur_log);
            
        }
    }
    else
    {
        sprintf(cur_log, "%s, 0, 0, 0, 0, 0, 0\n", image_file_name);
        log_buf_len += sprintf(log_buf + log_buf_len, "%s", cur_log);
        printf("%05d, %s", i, cur_log);
    }

}

static int forward_image(vision::Detector*     detector_interface,
                         int                   image_idx, 
                         const char*           image_file_dir,
                         const char*           image_file_name, 
                         char*                 log_buf,
                         long long int&        forward_cost)
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

    gettimeofday(&tv_start,0);
    res = detector_interface->run(raw_image, boxes_map);
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
                boxes_map, 
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
    static const int            LOG_BUF_SIZE          = (48 << 20);
    int                         log_offset            = 0;
    char*                       log_buf               = new char[LOG_BUF_SIZE];
    FILE*                       log_file              = 0;
    std::string                 log_file_name;
    long long int               forward_cost_min      = 0x7FFFFFFFFFFFFFFF;
    long long int               forward_cost_max      = 0;

    auto det     = new vision::Detector(cfg_file);
    std::vector<std::string>    image_list;

    image_list = travel_image_dir(img_file_dir, 
                                  std::vector<std::string>({std::string(".jpg"), std::string(".jpeg"), std::string(".bmp"), std::string(".png"), std::string(".pgm")}), 
                                  std::string(""));
    LOG(INFO) << "there " << (int)(image_list.size()) << " files in " << img_file_dir;
    if(0 == image_list.size())
        goto DLCV_DETECT_BATCH_LEAVE;
    memset(log_buf,        0, LOG_BUF_SIZE);

    if(0 == run_loop)
        run_loop = (int)(image_list.size());

    for(i = 0 ; i < run_loop; i ++)
    {
        cur_forward_cost  = 0;
        int cur_res       = forward_image(det,
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


DLCV_DETECT_BATCH_LEAVE:
    if(nullptr != log_buf)
        delete[] log_buf;

    return 0;
}
