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

static void normalize_image_uc1f1_naive(const unsigned char*  src,
                                        float*                dst,
                                        int                   width,
                                        int                   height,
                                        int                   src_line_size,
                                        int                   dst_line_size,
                                        float                 mean,
                                        float                 std)
{
    float inv_std      = 1.0f / std;
    int   line_ele_cnt = width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j ++)
        {
            dst[i * dst_line_size + j] = ((float)(src[i * src_line_size + j]) - mean) * inv_std;
        }
    }
}

static void  normalize_image_uc3f3_naive(const unsigned char*  src,
                                         float*                dst,
                                         int                   width,
                                         int                   height,
                                         int                   src_line_size,
                                         int                   dst_line_size,
                                         const float*          mean,
                                         const float*          std)
{
    float inv_std[3]   = { 1.0f / std[0], 1.0f / std[1], 1.0f / std[2] };
    int   line_ele_cnt = 3 * width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 3)
        {
            dst[i * dst_line_size + j]     = ((float)(src[i * src_line_size + j])     - mean[0]) * inv_std[0];
            dst[i * dst_line_size + j + 1] = ((float)(src[i * src_line_size + j + 1]) - mean[1]) * inv_std[1];
            dst[i * dst_line_size + j + 2] = ((float)(src[i * src_line_size + j + 2]) - mean[2]) * inv_std[2];
        }
    }
}

static void normalize_image_f1f1_naive(const float*  src,
                                       float*        dst,
                                       int           width,
                                       int           height,
                                       int           src_line_size,
                                       int           dst_line_size,
                                       float         mean,
                                       float         std)
{
    float inv_std      = 1.0f / std;
    int   line_ele_cnt = width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j ++)
        {
            dst[i * dst_line_size + j] = (src[i * src_line_size + j] - mean) * inv_std;
        }
    }
}

static void  normalize_image_f3f3_naive(const float*  src,
                                        float*        dst,
                                        int           width,
                                        int           height,
                                        int           src_line_size,
                                        int           dst_line_size,
                                        const float*  mean,
                                        const float*  std)
{
    float inv_std[3]   = { 1.0f / std[0], 1.0f / std[1], 1.0f / std[2] };
    int   line_ele_cnt = 3 * width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 3)
        {
            dst[i * dst_line_size + j]     = (src[i * src_line_size + j]     - mean[0]) * inv_std[0];
            dst[i * dst_line_size + j + 1] = (src[i * src_line_size + j + 1] - mean[1]) * inv_std[1];
            dst[i * dst_line_size + j + 2] = (src[i * src_line_size + j + 2] - mean[2]) * inv_std[2];
        }
    }
}

TEST(NormalizeImage, uc1f1) {
                  
    std::string          json_string = "{\n \"mean\": [ 123.675 ],\n\"std\": [ 58.395 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);

    int     res       = -1;
    int     i         = 0;
    int     width     = 1 + rand() % 2048;
    int     height    = 1 + rand() % 2048;
    int     channel   = 1;
    cv::Mat image_src(height, width, CV_8UC1);
    cv::Mat image_dst;
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = src_line_size;
    float*  dst_naive     = (float*)malloc(dst_line_size * height * sizeof(float));
    float   mean[]        = {
        123.675,
    };
    float   std[] = {
        58.395,
    };
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, height * dst_line_size * sizeof(float));

    normalize_image_uc1f1_naive((const unsigned char*)image_src.data,
                                dst_naive,
                                width, 
                                height,
                                src_line_size,
                                dst_line_size,
                                mean[0],
                                std[0]);

    vision::PreprocInfo info;
    normalize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        if(fabsf(dst_naive[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    free(dst_naive);
    EXPECT_EQ(res, 0);
}

TEST(NormalizeImage, uc3f3) 
{
    std::string          json_string = "{\n \"mean\": [ 123.675, 116.28, 103.53 ],\n\"std\": [ 58.395, 57.120003, 57.375 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);

    int     res           = -1;
    int     i             = 0;
    int     width         = 1 + rand() % 2048;
    int     height        = 1 + rand() % 2048;
    int     channel       = 3;
    cv::Mat image_src(height, width, CV_8UC3);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(unsigned char);
    int     dst_line_size = src_line_size;
    float*  dst_naive     = (float*)malloc(dst_line_size * height * sizeof(float));
    float   mean[]        = {
        123.675, 116.28, 103.53
    };
    float   std[]         = {
        58.395, 57.120003, 57.375
    };
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, dst_line_size * height * sizeof(float));

    normalize_image_uc3f3_naive((const unsigned char*)image_src.data,
                                dst_naive,
                                width, 
                                height,
                                src_line_size,
                                dst_line_size,
                                mean,
                                std);
    vision::PreprocInfo info;
    normalize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        if(fabsf(dst_naive[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    free(dst_naive);
    EXPECT_EQ(res, 0);
}

TEST(NormalizeImage, f1f1) {
                  
    std::string          json_string = "{\n \"mean\": [ 123.675 ],\n\"std\": [ 58.395 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);

    int     res       = -1;
    int     i         = 0;
    int     width     = 1 + rand() % 2048;
    int     height    = 1 + rand() % 2048;
    int     channel   = 1;
    cv::Mat image_src(height, width, CV_32FC1);
    cv::Mat image_dst;
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = src_line_size;
    float*  dst_naive     = (float*)malloc(src_line_size * height * sizeof(float));
    float   mean[]        = {
        123.675,
    };
    float   std[] = {
        58.395,
    };
    GenerateFloatData((float*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, height * dst_line_size * sizeof(float));

    normalize_image_f1f1_naive((const float*)image_src.data,
                               dst_naive,
                               width, 
                               height,
                               src_line_size,
                               dst_line_size,
                               mean[0],
                               std[0]);

    vision::PreprocInfo info;
    normalize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        if(fabsf(dst_naive[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    free(dst_naive);
    EXPECT_EQ(res, 0);
}

TEST(NormalizeImage, f3f3) 
{
    std::string          json_string = "{\n \"mean\": [ 123.675, 116.28, 103.53 ],\n\"std\": [ 58.395, 57.120003, 57.375 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);

    int     res           = -1;
    int     i             = 0;
    int     width         = 1 + rand() % 2048;
    int     height        = 1 + rand() % 2048;
    int     channel       = 3;
    cv::Mat image_src(height, width, CV_32FC3);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(float);
    int     dst_line_size = src_line_size;
    float*  dst_naive     = (float*)malloc(dst_line_size * height * sizeof(float));
    float   mean[]        = {
        123.675, 116.28, 103.53
    };
    float   std[]         = {
        58.395, 57.120003, 57.375
    };
    GenerateFloatData((float*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, dst_line_size * height * sizeof(float));

    normalize_image_f3f3_naive((const float*)image_src.data,
                               dst_naive,
                               width, 
                               height,
                               src_line_size,
                               dst_line_size,
                               mean,
                               std);
    vision::PreprocInfo info;
    normalize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        if(fabsf(dst_naive[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    free(dst_naive);
    EXPECT_EQ(res, 0);
}

TEST(NormalizeImage, performance_uc1f1)
{
    std::string          json_string = "{\n \"mean\": [ 123.675 ],\n\"std\": [ 58.395 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);
    int     res           = -1;
    int     i             = 0;
    int     width         = 640;
    int     height        = 480;
    int     channel       = 1;
    int     loop          = 10000;
    cv::Mat image_src(height, width, CV_8UC1);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(unsigned char);
    int     dst_line_size = src_line_size;
     vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc1f1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(NormalizeImage, performance_uc3f3)
{
    std::string          json_string = "{\n \"mean\": [ 123.675, 116.28, 103.53 ],\n\"std\": [ 58.395, 57.120003, 57.375 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);
    int     res           = -1;
    int     i             = 0;
    int     width         = 640;
    int     height        = 480;
    int     loop          = 10000;
    cv::Mat image_src(height, width, CV_8UC3);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(unsigned char);
    int     dst_line_size = src_line_size;
     vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc3f3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(NormalizeImage, performance_f1f1)
{
    std::string          json_string = "{\n \"mean\": [ 123.675 ],\n\"std\": [ 58.395 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);
    int     res           = -1;
    int     i             = 0;
    int     width         = 640;
    int     height        = 480;
    int     loop          = 10000;
    cv::Mat image_src(height, width, CV_32FC1);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(float);
    int     dst_line_size = src_line_size;
     vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateFloatData((float*)image_src.data, height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f1f1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(NormalizeImage, performance_f3f3)
{
    std::string          json_string = "{\n \"mean\": [ 123.675, 116.28, 103.53 ],\n\"std\": [ 58.395, 57.120003, 57.375 ] }";
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::NormalizeOp  normalize_op(json_patch);
    int     res           = -1;
    int     i             = 0;
    int     width         = 640;
    int     height        = 480;
    int     loop          = 10000;
    cv::Mat image_src(height, width, CV_32FC3);
    cv::Mat image_dst;
    int     src_line_size = image_src.step / sizeof(float);
    int     dst_line_size = src_line_size;
     vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateFloatData((float*)image_src.data, height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f3f3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 