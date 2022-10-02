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
#include <stdio.h>
#include <string.h>

static const float DIFF_EPS = 1e-1f;

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

TEST(ResizeImage, uc1) {
    srand(time(0));  

    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 1;
    cv::Mat image_src(src_height, src_width, CV_8UC1);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC1);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_8UC1);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    //FILE* f = fopen("/home/shaquille/WorkSpace/orion_workspace/inference_workspace/build/Debug/x86_64-linux/dlcv/test/resize_data.dat", "rb");
    //FILE* f = fopen("resize_data.dat", "rb");
    //fread(((unsigned char*)(image_src.data)),  src_height * src_line_size, 1, f);
    //fflush(f);
    //fclose(f);

    memset((unsigned char*)image_dst_naive.data,   0, dst_height * dst_line_size);
    memset((unsigned char*)image_dst.data,         0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0., cv::INTER_LINEAR);
/*
    FILE* f = fopen("./resize_data.dat", "wb");
    fwrite(((unsigned char*)(image_src.data)),  src_height * src_line_size, 1, f);
    fflush(f);
    fclose(f);
*/
    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);
/*
    printf("%d, %d, %d, %d\n",
           ((unsigned char*)(image_src.data))[0],
           ((unsigned char*)(image_src.data))[1],
           ((unsigned char*)(image_src.data))[src_line_size],
           ((unsigned char*)(image_src.data))[src_line_size + 1]);

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
           ((unsigned char*)(image_dst.data))[0],
           ((unsigned char*)(image_dst.data))[1],
           ((unsigned char*)(image_dst.data))[2],
           ((unsigned char*)(image_dst.data))[3],
           ((unsigned char*)(image_dst.data))[4],
           ((unsigned char*)(image_dst.data))[5],
           ((unsigned char*)(image_dst.data))[6],
           ((unsigned char*)(image_dst.data))[7],
           ((unsigned char*)(image_dst.data))[8],
           ((unsigned char*)(image_dst.data))[9],
           ((unsigned char*)(image_dst.data))[10],
           ((unsigned char*)(image_dst.data))[11],
           ((unsigned char*)(image_dst.data))[12],
           ((unsigned char*)(image_dst.data))[13],
           ((unsigned char*)(image_dst.data))[14],
           ((unsigned char*)(image_dst.data))[15]);
*/
    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        int  dst_naive_val = ((unsigned char*)image_dst_naive.data)[i];
        int  dst_val       = ((unsigned char*)image_dst.data)[i];
        //if(abs(dst_naive_val-dst_val) > 1)
        //if(dst_naive_val != dst_val)
        if(abs(dst_naive_val-dst_val) > 1)
        {
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            res ++;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, uc3) 
{
    srand(time(0));
    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 3;
    cv::Mat image_src(src_height, src_width, CV_8UC3);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC3);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_8UC3);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    memset((unsigned char*)image_dst_naive.data,   0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0, cv::INTER_LINEAR);

    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        if(abs(((unsigned char*)image_dst_naive.data)[i]-((unsigned char*)image_dst.data)[i]) > 1)
        {
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            res += 1;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, uc4) 
{
    srand(time(0));
    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 4;
    cv::Mat image_src(src_height, src_width, CV_8UC4);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC4);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_8UC4);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    memset((unsigned char*)image_dst_naive.data,   0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0, cv::INTER_LINEAR);

    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        if(abs(((unsigned char*)image_dst_naive.data)[i]-((unsigned char*)image_dst.data)[i]) > 1)
        {
            printf("error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_dst_naive.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, f1) {   
    srand(time(0));            
    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 1;
    cv::Mat image_src(src_height, src_width, CV_32FC1);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC1);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_32FC1);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    memset((float*)image_dst_naive.data,   0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0.0, cv::INTER_LINEAR);

    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        if(fabsf(((float*)image_dst_naive.data)[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", 
                   i, 
                   ((float*)image_dst_naive.data)[i], 
                   ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, f3) 
{
    srand(time(0));
    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 3;
    cv::Mat image_src(src_height, src_width, CV_32FC3);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC3);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_32FC3);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    memset((float*)image_dst_naive.data,   0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0, cv::INTER_LINEAR);

    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        if(fabsf(((float*)image_dst_naive.data)[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", 
                   i, 
                   ((float*)image_dst_naive.data)[i], 
                   ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, f4)
{
    srand(time(0));
    int     dst_width  = 1 + rand() % 2048;
    int     dst_height = 1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1 + rand() % 2048;
    int     src_height = 1 + rand() % 2048;
    int     channel    = 4;
    cv::Mat image_src(src_height, src_width, CV_32FC4);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC4);
    cv::Mat image_dst_naive(dst_height, dst_width, CV_32FC4);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    memset((float*)image_dst_naive.data,   0, dst_height * dst_line_size);

    printf("src: %d, %d\n", src_width, src_height);
    printf("dst: %d, %d\n", dst_width, dst_height);

    cv::resize(image_src, image_dst_naive, cv::Size(dst_width, dst_height), 0., 0, cv::INTER_LINEAR);

    vision::PreprocInfo info;
    resize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * dst_height) ; i ++)
    {
        if(fabsf(((float*)image_dst_naive.data)[i]-((float*)image_dst.data)[i]) > DIFF_EPS)
        {
            printf("error, %d, %.6f, %.6f\n", 
                   i, 
                   ((float*)image_dst_naive.data)[i], 
                   ((float*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }

    EXPECT_EQ(res, 0);
}

TEST(ResizeImage, performance_uc1)
{
    int     dst_width  = 640;
    int     dst_height = 480;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;
    int     src_height = 720;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_8UC1);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC1);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(ResizeImage, performance_uc3)
{
    int     dst_width  = 640;
    int     dst_height = 480;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;
    int     src_height = 720;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_8UC3);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC3);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(ResizeImage, performance_uc4)
{
    int     dst_width  = 640;
    int     dst_height = 480;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;
    int     src_height = 720;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_8UC4);
    cv::Mat image_dst(dst_height, dst_width, CV_8UC4);
    int     src_line_size = image_src.step[0] / sizeof(unsigned char);
    int     dst_line_size = image_dst.step[0] / sizeof(unsigned char);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateUCData((unsigned char*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(ResizeImage, performance_f1)
{
    int     dst_width  = 640;//1 + rand() % 2048;
    int     dst_height = 480;//1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;//1 + rand() % 2048;
    int     src_height = 720; //1 + rand() % 2048;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_32FC1);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC1);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(ResizeImage, performance_f3)
{
    int     dst_width  = 640;//1 + rand() % 2048;
    int     dst_height = 480;//1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;//1 + rand() % 2048;
    int     src_height = 720; //1 + rand() % 2048;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_32FC3);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC3);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(ResizeImage, performance_f4)
{
    int     dst_width  = 640;//1 + rand() % 2048;
    int     dst_height = 480;//1 + rand() % 2048;
    std::string          json_string = std::string("{\n \"width\": ") + std::to_string(dst_width) + 
                                       std::string(",\n\"height\": ") + std::to_string(dst_height) +
                                       std::string(",\n\"mode\":\"bilinear\" }");
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::ResizeOp     resize_op(json_patch);

    int     res        = -1;
    int     i          = 0;
    int     src_width  = 1280;//1 + rand() % 2048;
    int     src_height = 720; //1 + rand() % 2048;
    int     loop       = 1000;
    cv::Mat image_src(src_height, src_width, CV_32FC4);
    cv::Mat image_dst(dst_height, dst_width, CV_32FC4);
    int     src_line_size = image_src.step[0] / sizeof(float);
    int     dst_line_size = image_dst.step[0] / sizeof(float);

    vision::PreprocInfo info;

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    GenerateFloatData((float*)image_src.data, src_height * src_line_size);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = resize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f4, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 