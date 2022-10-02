#include <cstdint>              // for uint8_t
#include <memory>               // for shared_ptr
#include <tuple>                // for get
#include <typeinfo>             // for typeid
#include <typeindex>            // for type_index
#include <utility>              // for move
#include <vector>               // for vector
#include <gtest/gtest.h>
#include "../src/preprocess/preprocess.h"
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

static const float DIFF_EPS = 1e-6f;

static int g_iRandInit = 0;
static void GenerateFloatData(float* dst, int data_cnt)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }

    for(i = 0 ; i < data_cnt ; i ++)
        dst[i] = (float)(rand()%256);
}

static void GenerateUCData(unsigned char* dst, int data_cnt)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }

    for(i = 0 ; i < data_cnt ; i ++)
        dst[i] = (unsigned char)(rand()%256);
}

TEST(image_togray, uc3_bgr) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_8UC3);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_BGR2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            unsigned int d0 = ((unsigned char*)(image_src.data))[3 * i];
            unsigned int d1 = ((unsigned char*)(image_src.data))[3 * i + 1];
            unsigned int d2 = ((unsigned char*)(image_src.data))[3 * i + 2];
            float        d  = d0 * 0.114f + d1 * 0.587f + d2 * 0.299f;
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            printf("src %d, %d, %d, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, uc3_rgb) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"rgb2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_8UC3);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_RGB2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            unsigned int d0 = ((unsigned char*)(image_src.data))[3 * i];
            unsigned int d1 = ((unsigned char*)(image_src.data))[3 * i + 1];
            unsigned int d2 = ((unsigned char*)(image_src.data))[3 * i + 2];
            float        d  = d2 * 0.114f + d1 * 0.587f + d0 * 0.299f;
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            printf("src %d, %d, %d, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, uc4_bgr) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_8UC4);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_BGR2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            unsigned int d0 = ((unsigned char*)(image_src.data))[4 * i];
            unsigned int d1 = ((unsigned char*)(image_src.data))[4 * i + 1];
            unsigned int d2 = ((unsigned char*)(image_src.data))[4 * i + 2];
            float        d  = d0 * 0.114f + d1 * 0.587f + d2 * 0.299f;
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            printf("src %d, %d, %d, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, uc4_rgb) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"rgb2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_8UC4);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_RGB2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            unsigned int d0 = ((unsigned char*)(image_src.data))[4 * i];
            unsigned int d1 = ((unsigned char*)(image_src.data))[4 * i + 1];
            unsigned int d2 = ((unsigned char*)(image_src.data))[4 * i + 2];
            float        d  = d2 * 0.114f + d1 * 0.587f + d0 * 0.299f;
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            printf("src %d, %d, %d, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, f3_bgr) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    cv::Mat         image_src(height, width, CV_32FC3);
    cv::Mat         image_dst_naive(height, width, CV_32FC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_BGR2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    int width_aligned = ((width >> 3) << 3);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        float a = ((float*)image_dst_naive.data)[i];
        float b = ((float*)image_dst.data)[i];
        int   x = (i % width);
        if(((x >= width_aligned) && fabs(a - b) > 1e-4f) ||
           ((x < width_aligned) && fabs(a - b) > DIFF_EPS))
        {
            float d0 = ((float*)(image_src.data))[3 * i];
            float d1 = ((float*)(image_src.data))[3 * i + 1];
            float d2 = ((float*)(image_src.data))[3 * i + 2];
            float d  = (d1 * 0.587f + d0 * 0.114f + d2 * 0.299f);
            {
                printf("error, %d, %.6f, %.6f\n", 
                    i, 
                    ((float*)image_dst_naive.data)[i], 
                    ((float*)image_dst.data)[i]);
                printf("src %.6f, %.6f, %.6f, %.6f\n", d0, d1, d2, d);
                res =-1;
            }
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, f3_rgb) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"rgb2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    cv::Mat         image_src(height, width, CV_32FC3);
    cv::Mat         image_dst_naive(height, width, CV_32FC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_RGB2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    int width_aligned = ((width >> 3) << 3);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        float a = ((float*)image_dst_naive.data)[i];
        float b = ((float*)image_dst.data)[i];
        int   x = (i % width);
        if(((x >= width_aligned) && fabs(a - b) > 1e-4f) ||
           ((x < width_aligned) && fabs(a - b) > DIFF_EPS))
        {
            float d0 = ((float*)(image_src.data))[3 * i];
            float d1 = ((float*)(image_src.data))[3 * i + 1];
            float d2 = ((float*)(image_src.data))[3 * i + 2];
            float d  = (d1 * 0.587f + d0 * 0.114f + d2 * 0.299f);
            {
                printf("error, %d, %.6f, %.6f\n", 
                    i, 
                    ((float*)image_dst_naive.data)[i], 
                    ((float*)image_dst.data)[i]);
                printf("src %.6f, %.6f, %.6f, %.6f\n", d0, d1, d2, d);
                res =-1;
            }
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, f4_bgr) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    cv::Mat         image_src(height, width, CV_32FC4);
    cv::Mat         image_dst_naive(height, width, CV_32FC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_BGR2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    int width_aligned = ((width >> 3) << 3);
    
    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        float a = ((float*)image_dst_naive.data)[i];
        float b = ((float*)image_dst.data)[i];
        int   x = (i % width);
        if(((x >= width_aligned) && fabs(a - b) > 1e-4f) ||
           ((x < width_aligned) && fabs(a - b) > DIFF_EPS))
        {
            float d0 = ((float*)(image_src.data))[4 * i];
            float d1 = ((float*)(image_src.data))[4 * i + 1];
            float d2 = ((float*)(image_src.data))[4 * i + 2];
            float d  = (d1 * 0.587f + d0 * 0.114f + d2 * 0.299f);
            printf("error, %d, %.6f, %.6f\n", 
                i, 
                ((float*)image_dst_naive.data)[i], 
                ((float*)image_dst.data)[i]);
            printf("src %.6f, %.6f, %.6f, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, f4_rgb) {
    srand(time(0));
    std::string          json_string = "{\n \"code\": \"rgb2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    cv::Mat         image_src(height, width, CV_32FC4);
    cv::Mat         image_dst_naive(height, width, CV_32FC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    cv::cvtColor(image_src, image_dst_naive, cv::COLOR_RGB2GRAY);

    printf("src : %d, %d\n", width, height);

    vision::PreprocInfo info;
    togray_op.run(image_src, info, image_dst);

    int width_aligned = ((width >> 3) << 3);
    
    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        float a = ((float*)image_dst_naive.data)[i];
        float b = ((float*)image_dst.data)[i];
        int   x = (i % width);
        if(((x >= width_aligned) && fabs(a - b) > 1e-4f) ||
           ((x < width_aligned) && fabs(a - b) > DIFF_EPS))
        {
            float d0 = ((float*)(image_src.data))[4 * i];
            float d1 = ((float*)(image_src.data))[4 * i + 1];
            float d2 = ((float*)(image_src.data))[4 * i + 2];
            float d  = (d1 * 0.587f + d0 * 0.114f + d2 * 0.299f);
            printf("error, %d, %.6f, %.6f\n", 
                i, 
                ((float*)image_dst_naive.data)[i], 
                ((float*)image_dst.data)[i]);
            printf("src %.6f, %.6f, %.6f, %.6f\n", d0, d1, d2, d);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_togray, performance_uc3_bgr)
{
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             loop      = 1000;
    int             res       = -1;
    int             i         = 0;
    int             width     = 640;
    int             height    = 480;
    cv::Mat         image_src(height, width, CV_8UC3);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    vision::PreprocInfo info;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = togray_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc3_bgr, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 


TEST(image_togray, performance_uc4_bgr)
{
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             loop      = 1000;
    int             res       = -1;
    int             i         = 0;
    int             width     = 640;
    int             height    = 480;
    cv::Mat         image_src(height, width, CV_8UC4);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);

    vision::PreprocInfo info;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = togray_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc4_bgr, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(image_togray, performance_f3_bgr)
{
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             loop      = 1000;
    int             res       = -1;
    int             i         = 0;
    int             width     = 640;
    int             height    = 480;
    cv::Mat         image_src(height, width, CV_32FC3);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    vision::PreprocInfo info;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = togray_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f3_bgr, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(image_togray, performance_f4_bgr)
{
    std::string          json_string = "{\n \"code\": \"bgr2gray\" \n}";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ToGrayOp     togray_op(json_patch);

    int             loop      = 1000;
    int             res       = -1;
    int             i         = 0;
    int             width     = 640;
    int             height    = 480;
    cv::Mat         image_src(height, width, CV_32FC4);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    vision::PreprocInfo info;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = togray_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f4_bgr, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 