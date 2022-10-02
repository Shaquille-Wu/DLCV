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

static void batch_normalize_image_uc1f1_naive(const unsigned char*  src,
                                              float*                dst,
                                              int                   width,
                                              int                   height,
                                              int                   src_line_size,
                                              int                   dst_line_size,
                                              float                 beta,
                                              float                 mean,
                                              float                 gamma,
                                              float                 var)
{
    int    line_ele_cnt = width;
    double bn_mean      = mean;
    double bn_var       = 1 / sqrt(var + 1e-5);
    double bn_gamma     = gamma;
    double bn_beta      = beta;
    bn_gamma            = bn_gamma * bn_var;

    float  f_mean       = bn_mean;
    float  f_gamma      = bn_gamma;
    float  f_beta       = bn_beta;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j ++)
        {
            dst[i * dst_line_size + j] = ((float)(src[i * src_line_size + j]) - f_mean) * f_gamma + f_beta;
        }
    }
}

static void  batch_normalize_image_uc3f3_naive(const unsigned char*  src,
                                               float*                dst,
                                               int                   width,
                                               int                   height,
                                               int                   src_line_size,
                                               int                   dst_line_size,
                                               const float*          beta,
                                               const float*          mean,
                                               const float*          gamma,
                                               const float*          var)
{
    double bn_mean0      = mean[0];
    double bn_var0       = 1 / sqrt(var[0] + 1e-5);
    double bn_gamma0     = gamma[0];
    double bn_beta0      = beta[0];
    bn_gamma0            = bn_gamma0 * bn_var0;

    double bn_mean1      = mean[1];
    double bn_var1       = 1 / sqrt(var[1] + 1e-5);
    double bn_gamma1     = gamma[1];
    double bn_beta1      = beta[1];
    bn_gamma1            = bn_gamma1 * bn_var1;

    double bn_mean2      = mean[2];
    double bn_var2       = 1 / sqrt(var[2] + 1e-5);
    double bn_gamma2     = gamma[2];
    double bn_beta2      = beta[2];
    bn_gamma2            = bn_gamma2 * bn_var2;
    float  f_mean0       = bn_mean0;
    float  f_gamma0      = bn_gamma0;
    float  f_beta0       = bn_beta0;
    float  f_mean1       = bn_mean1;
    float  f_gamma1      = bn_gamma1;
    float  f_beta1       = bn_beta1;
    float  f_mean2       = bn_mean2;
    float  f_gamma2      = bn_gamma2;
    float  f_beta2       = bn_beta2;

    int   line_ele_cnt = 3 * width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 3)
        {
            dst[i * dst_line_size + j]     = ((float)(src[i * src_line_size + j])     - f_mean0) * f_gamma0 + f_beta0;
            dst[i * dst_line_size + j + 1] = ((float)(src[i * src_line_size + j + 1]) - f_mean1) * f_gamma1 + f_beta1;
            dst[i * dst_line_size + j + 2] = ((float)(src[i * src_line_size + j + 2]) - f_mean2) * f_gamma2 + f_beta2;
        }
    }
}

static void batch_normalize_image_f1f1_naive(const float*  src,
                                             float*        dst,
                                             int           width,
                                             int           height,
                                             int           src_line_size,
                                             int           dst_line_size,
                                             float         beta,
                                             float         mean,
                                             float         gamma,
                                             float         var)
{
    int    line_ele_cnt = width;
    double bn_mean      = mean;
    double bn_var       = 1 / sqrt(var + 1e-5);
    double bn_gamma     = gamma;
    double bn_beta      = beta;
    bn_gamma            = bn_gamma * bn_var;

    float  f_mean       = bn_mean;
    float  f_gamma      = bn_gamma;
    float  f_beta       = bn_beta;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j ++)
        {
            dst[i * dst_line_size + j] = (src[i * src_line_size + j] - f_mean) * f_gamma + f_beta;
        }
    }
}

static void  batch_normalize_image_f3f3_naive(const float*          src,
                                              float*                dst,
                                              int                   width,
                                              int                   height,
                                              int                   src_line_size,
                                              int                   dst_line_size,
                                              const float*          beta,
                                              const float*          mean,
                                              const float*          gamma,
                                              const float*          var)
{
    int   line_ele_cnt   = 3 * width;
    double bn_mean0      = mean[0];
    double bn_var0       = 1 / sqrt(var[0] + 1e-5);
    double bn_gamma0     = gamma[0];
    double bn_beta0      = beta[0];
    bn_gamma0            = bn_gamma0 * bn_var0;

    double bn_mean1      = mean[1];
    double bn_var1       = 1 / sqrt(var[1] + 1e-5);
    double bn_gamma1     = gamma[1];
    double bn_beta1      = beta[1];
    bn_gamma1            = bn_gamma1 * bn_var1;

    double bn_mean2      = mean[2];
    double bn_var2       = 1 / sqrt(var[2] + 1e-5);
    double bn_gamma2     = gamma[2];
    double bn_beta2      = beta[2];
    bn_gamma2            = bn_gamma2 * bn_var2;
    float  f_mean0       = bn_mean0;
    float  f_gamma0      = bn_gamma0;
    float  f_beta0       = bn_beta0;
    float  f_mean1       = bn_mean1;
    float  f_gamma1      = bn_gamma1;
    float  f_beta1       = bn_beta1;
    float  f_mean2       = bn_mean2;
    float  f_gamma2      = bn_gamma2;
    float  f_beta2       = bn_beta2;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 3)
        {
            dst[i * dst_line_size + j]     = (src[i * src_line_size + j]     - f_mean0) * f_gamma0 + f_beta0;
            dst[i * dst_line_size + j + 1] = (src[i * src_line_size + j + 1] - f_mean1) * f_gamma1 + f_beta1;
            dst[i * dst_line_size + j + 2] = (src[i * src_line_size + j + 2] - f_mean2) * f_gamma2 + f_beta2;
        }
    }
}

TEST(BatchNormalizeImage, uc1f1) {
    std::string               json_string = "{\n \"beta\": [ 3.12500215 ],\n\"mean\": [ 103.65445709 ],\n\"gamma\": [ 0.16505176 ], \n\"var\": [ 3049.10253906 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);

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
    float   beta[]        = {
        3.12500215,
    };
    float   mean[]        = {
        103.65445709,
    };
    float   gamma[]       = {
        0.16505176,
    };
    float   var[]         = {
        3049.10253906,
    };
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, height * dst_line_size * sizeof(float));

    batch_normalize_image_uc1f1_naive((const unsigned char*)image_src.data,
                                       dst_naive,
                                       width, 
                                       height,
                                       src_line_size,
                                       dst_line_size,
                                       beta[0],
                                       mean[0],
                                       gamma[0],
                                       var[0]);

    vision::PreprocInfo info;
    batch_normalize_op.run(image_src, info, image_dst);

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

TEST(BatchNormalizeImage, uc3f3) 
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215,-0.97326922,-0.30886102 ],\n\"mean\": [ 103.65445709, 121.083992, 158.1137085],\n\"gamma\": [ 0.16505176, 0.60845631, 0.59119201], \n\"var\": [ 3049.10253906, 3112.63256836, 3859.73632812 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);

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
    float   beta[]        = {
        3.12500215,
        -0.97326922,
        -0.30886102
    };
    float   mean[]        = {
        103.65445709,
        121.083992,
        158.1137085
    };
    float   gamma[]       = {
        0.16505176,
        0.60845631,
        0.59119201
    };
    float   var[]         = {
        3049.10253906,
        3112.63256836,
        3859.73632812
    };
    GenerateUCData((unsigned char*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, dst_line_size * height * sizeof(float));

    batch_normalize_image_uc3f3_naive((const unsigned char*)image_src.data,
                                      dst_naive,
                                      width, 
                                      height,
                                      src_line_size,
                                      dst_line_size,
                                      beta,
                                      mean,
                                      gamma,
                                      var);
    vision::PreprocInfo info;
    batch_normalize_op.run(image_src, info, image_dst);

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

TEST(BatchNormalizeImage, f1f1) {
                  
    std::string               json_string = "{\n \"beta\": [ 3.12500215 ],\n\"mean\": [ 103.65445709 ],\n\"gamma\": [ 0.16505176 ], \n\"var\": [ 3049.10253906 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);

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
    float   beta[]        = {
        3.12500215,
    };
    float   mean[]        = {
        103.65445709,
    };
    float   gamma[]       = {
        0.16505176,
    };
    float   var[]         = {
        3049.10253906,
    };
    GenerateFloatData((float*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, height * dst_line_size * sizeof(float));

    batch_normalize_image_f1f1_naive((const float*)image_src.data,
                                     dst_naive,
                                     width, 
                                     height,
                                     src_line_size,
                                     dst_line_size,
                                     beta[0],
                                     mean[0],
                                     gamma[0],
                                     var[0]);

    vision::PreprocInfo info;
    batch_normalize_op.run(image_src, info, image_dst);

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

TEST(BatchNormalizeImage, f3f3) 
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215,-0.97326922,-0.30886102 ],\n\"mean\": [ 103.65445709, 121.083992, 158.1137085],\n\"gamma\": [ 0.16505176, 0.60845631, 0.59119201], \n\"var\": [ 3049.10253906, 3112.63256836, 3859.73632812 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);

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
    float   beta[]        = {
        3.12500215,
        -0.97326922,
        -0.30886102
    };
    float   mean[]        = {
        103.65445709,
        121.083992,
        158.1137085
    };
    float   gamma[]       = {
        0.16505176,
        0.60845631,
        0.59119201
    };
    float   var[]         = {
        3049.10253906,
        3112.63256836,
        3859.73632812
    };
    GenerateFloatData((float*)image_src.data, height * src_line_size);
    memset(dst_naive,      0, dst_line_size * height * sizeof(float));

    batch_normalize_image_f3f3_naive((const float*)image_src.data,
                                     dst_naive,
                                     width, 
                                     height,
                                     src_line_size,
                                     dst_line_size,
                                     beta,
                                     mean,
                                     gamma,
                                     var);
    vision::PreprocInfo info;
    batch_normalize_op.run(image_src, info, image_dst);

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

TEST(BatchNormalizeImage, performance_uc1f1)
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215 ],\n\"mean\": [ 103.65445709 ],\n\"gamma\": [ 0.16505176 ], \n\"var\": [ 3049.10253906 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);
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
        res = batch_normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc1f1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(BatchNormalizeImage, performance_uc3f3)
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215,-0.97326922,-0.30886102 ],\n\"mean\": [ 103.65445709, 121.083992, 158.1137085],\n\"gamma\": [ 0.16505176, 0.60845631, 0.59119201], \n\"var\": [ 3049.10253906, 3112.63256836, 3859.73632812 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);
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
        res = batch_normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_uc3f3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(BatchNormalizeImage, performance_f1f1)
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215 ],\n\"mean\": [ 103.65445709 ],\n\"gamma\": [ 0.16505176 ], \n\"var\": [ 3049.10253906 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);
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
        res = batch_normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f1f1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 

TEST(BatchNormalizeImage, performance_f3f3)
{
    std::string               json_string = "{\n \"beta\": [ 3.12500215,-0.97326922,-0.30886102 ],\n\"mean\": [ 103.65445709, 121.083992, 158.1137085],\n\"gamma\": [ 0.16505176, 0.60845631, 0.59119201], \n\"var\": [ 3049.10253906, 3112.63256836, 3859.73632812 ] }";
    nlohmann::json            json_patch  = json::parse(json_string);
    vision::BatchNormOp       batch_normalize_op(json_patch);
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
        res = batch_normalize_op.run(image_src, info, image_dst);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_f3f3, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);
} 