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

static void GenerateFloatData(float* dst, int data_cnt)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }

    for(i = 0 ; i < data_cnt ; i ++)
        dst[i] = ((float)(rand() % 6)) * (float)(rand()%256) / 256.0f;
}

static void ToInt8Naive_f1uc1(const float*     src,
                              unsigned char*   dst,
                              int              width,
                              int              height,
                              int              src_line_size,
                              int              dst_line_size,
                              float            step,
                              float            zero)
{
    float  scale = 1.0f / step;
    for(int i = 0 ; i < height ; i ++)
    {
        for(int j = 0; j < width ; j ++)
        {
            float   a = src[i * src_line_size + j];
            float   b = a * scale + zero;
            int     c = b < 0.0f ? b - 0.5f : b + 0.5f;
            int     d = c < 0 ? 0 : (c > 255 ? 255 : c);
            dst[i * dst_line_size + j] = (unsigned char)d;
        }
    }
}

static void ToInt8Naive_f3uc3(const float*     src,
                              unsigned char*   dst,
                              int              width,
                              int              height,
                              int              src_line_size,
                              int              dst_line_size,
                              float            step,
                              float            zero)
{
    float  scale = 1.0f / step;
    for(int i = 0 ; i < height ; i ++)
    {
        for(int j = 0; j < width ; j ++)
        {
            float   a0 = src[i * src_line_size + 3 * j];
            float   a1 = src[i * src_line_size + 3 * j + 1];
            float   a2 = src[i * src_line_size + 3 * j + 2];
            float   b0 = a0 * scale + zero;
            float   b1 = a1 * scale + zero;
            float   b2 = a2 * scale + zero;
            int     c0 = b0 < 0.0f ? b0 - 0.5f : b0 + 0.5f;
            int     c1 = b1 < 0.0f ? b1 - 0.5f : b1 + 0.5f;
            int     c2 = b2 < 0.0f ? b2 - 0.5f : b2 + 0.5f;
            int     d0 = c0 < 0 ? 0 : (c0 > 255 ? 255 : c0);
            int     d1 = c1 < 0 ? 0 : (c1 > 255 ? 255 : c1);
            int     d2 = c2 < 0 ? 0 : (c2 > 255 ? 255 : c2);
            dst[i * dst_line_size + 3 * j]     = (unsigned char)d0;
            dst[i * dst_line_size + 3 * j + 1] = (unsigned char)d1;
            dst[i * dst_line_size + 3 * j + 2] = (unsigned char)d2;
        }
    }
}

TEST(image_toint8, f1uc1) {

    srand(time(0));

    float                step         = (((float)(rand() % 10)) * (float)(rand() % 4096) / 4096.0f) + 0.001f;
    float                zero         = (float)(rand() % 128);
    char                 sz_json[256] = { 0 };

    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch   = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_32FC1);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    ToInt8Naive_f1uc1((float*)image_src.data, (unsigned char*)(image_dst_naive.data), width, height, src_line_size, dst_line_size, step, zero);

    printf("src : %d, %d, step: %.8f, zero: %.8f\n", width, height, step, zero);

    vision::PreprocInfo info;
    quantize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
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

TEST(image_toint8, normalize_f1uc1) {

    srand(time(0));

    float                step         = (((float)(rand() % 10)) * (float)(rand() % 4096) / 4096.0f) + 0.001f;
    float                zero         = (float)(rand() % 128);
    char                 sz_json[256] = { 0 };

    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch   = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    float           norm_std   = 1.0f / step;
    float           norm_mean  = zero;
    sprintf(sz_json, "{\n \"mean\": [%.8f], \n\"std\": [%.8f]\n}", norm_mean, norm_std);
    json_string  = std::string(sz_json);
    nlohmann::json       norm_json_patch   = json::parse(json_string);
    vision::NormalizeOp  normalize_op(norm_json_patch);

    int             res        = -1;
    int             i          = 0;
    int             width      = 1 + rand() % 2048;
    int             height     = 1 + rand() % 2048;
    int             channel    = 1;
    cv::Mat         image_raw(height, width, CV_8UC1);
    cv::Mat         image_normalize(height, width, CV_32FC1);
    int             raw_line_size  = image_raw.step[0];
    int             norm_line_size = image_normalize.step[0] / sizeof(float);
    cv::Mat         image_dst_naive(height, width, CV_8UC1);
    cv::Mat         image_dst;
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_raw.data, height * raw_line_size);
    
    vision::PreprocInfo info;
    normalize_op.run(image_raw, info, image_normalize);

    ToInt8Naive_f1uc1((float*)image_normalize.data, (unsigned char*)(image_dst_naive.data), width, height, norm_line_size, dst_line_size, step, zero);

    printf("src : %d, %d, step: %.8f, zero: %.8f\n", width, height, step, zero);

    vision::PreprocInfo quant_info;
    quantize_op.run(image_normalize, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
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

    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_raw.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            printf("raw error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_raw.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_toint8, f3uc3) {
    srand(time(0));
    float                step         = (((float)(rand() % 10)) * (float)(rand() % 4096) / 4096.0f) + 0.001f;
    float                zero         = (float)(rand() % 128);
    char                 sz_json[256] = { 0 };
    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch   = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + rand() % 2048;
    int             height    = 1 + rand() % 2048;
    int             channel   = 1;
    cv::Mat         image_src(height, width, CV_32FC3);
    cv::Mat         image_dst_naive(height, width, CV_8UC3);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    ToInt8Naive_f3uc3((float*)image_src.data, (unsigned char*)image_dst_naive.data, width, height, src_line_size, dst_line_size, step, zero);

    printf("src : %d, %d, step: %.8f, zero: %.8f\n", width, height, step, zero);

    vision::PreprocInfo info;
    quantize_op.run(image_src, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
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

TEST(image_toint8, normalize_f3uc3) {

    srand(time(0));

    float                step         = (((float)(rand() % 10)) * (float)(rand() % 4096) / 4096.0f) + 0.001f;
    float                zero         = (float)(rand() % 128);
    char                 sz_json[256] = { 0 };

    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch   = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    float           norm_std   = 1.0f / step;
    float           norm_mean  = zero;
    sprintf(sz_json, "{\n \"mean\": [%.8f, %.8f, %.8f], \n\"std\": [%.8f, %.8f, %.8f]\n}", norm_mean, norm_mean, norm_mean, norm_std, norm_std, norm_std);
    json_string  = std::string(sz_json);
    nlohmann::json       norm_json_patch   = json::parse(json_string);
    vision::NormalizeOp  normalize_op(norm_json_patch);

    int             res        = -1;
    int             i          = 0;
    int             width      = 1 + rand() % 2048;
    int             height     = 1 + rand() % 2048;
    int             channel    = 1;
    cv::Mat         image_raw(height, width, CV_8UC3);
    cv::Mat         image_normalize(height, width, CV_32FC3);
    int             raw_line_size  = image_raw.step[0];
    int             norm_line_size = image_normalize.step[0] / sizeof(float);
    cv::Mat         image_dst_naive(height, width, CV_8UC3);
    cv::Mat         image_dst;
    int             dst_line_size = image_dst_naive.step[0] / sizeof(unsigned char);
    GenerateUCData((unsigned char*)image_raw.data, height * raw_line_size);
    
    vision::PreprocInfo info;
    normalize_op.run(image_raw, info, image_normalize);

    ToInt8Naive_f3uc3((float*)image_normalize.data, (unsigned char*)(image_dst_naive.data), width, height, norm_line_size, dst_line_size, step, zero);

    printf("src : %d, %d, step: %.8f, zero: %.8f\n", width, height, step, zero);

    vision::PreprocInfo quant_info;
    quantize_op.run(image_normalize, info, image_dst);

    res = 0;
    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_dst_naive.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
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

    for(i = 0 ; i < (dst_line_size * height) ; i ++)
    {
        unsigned char a = ((unsigned char*)image_raw.data)[i];
        unsigned char b = ((unsigned char*)image_dst.data)[i];
        if(a != b)
        {
            printf("raw error, %d, %d, %d\n", 
                   i, 
                   ((unsigned char*)image_raw.data)[i], 
                   ((unsigned char*)image_dst.data)[i]);
            res =-1;
            break;
        }
    }
    EXPECT_EQ(res, 0);
}

TEST(image_toint8, performance_f1uc1)
{
    float                step         = 0.02968111225202972f;
    float                zero         = 0.0f;
    char                 sz_json[256] = { 0 };
    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch   = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    int             loop      = 10000;
    int             res       = -1;
    int             i         = 0;
    int             width     = 640;
    int             height    = 480;
    cv::Mat         image_src(height, width, CV_32FC1);
    cv::Mat         image_dst;
    int             src_line_size = image_src.step[0] / sizeof(float);
    GenerateFloatData((float*)image_src.data, height * src_line_size);

    vision::PreprocInfo info;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        res = quantize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f1uc1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 


TEST(image_toint8, performance_f3uc3)
{
    float                step         = 0.02968111225202972f;
    float                zero         = 0.0f;
    char                 sz_json[256] = { 0 };
    sprintf(sz_json, "{\n \"step\": %.8f, \n\"zero\": %.8f\n}", step, zero);
    std::string          json_string  = std::string(sz_json);
    nlohmann::json       json_patch  = json::parse(json_string);
    vision::QuantizeOp   quantize_op(json_patch);

    int             loop      = 10000;
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
        res = quantize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f3uc3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 