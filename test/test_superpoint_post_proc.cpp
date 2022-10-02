#include <cstdint>              // for uint8_t
#include <memory>               // for shared_ptr
#include <tuple>                // for get
#include <typeinfo>             // for typeid
#include <typeindex>            // for type_index
#include <utility>              // for move
#include <vector>               // for vector
//#include <cstdlib>
#include <gtest/gtest.h>
#include "../dlcv_proc_opt/dlcv_proc_opt.h"
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/preprocess/preprocess.h"

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
        dst[i] = (float)(rand()%256) / 255.0f;
}

/*
static int cmp_xy(const void* left, const void* right)
{
    DLCV_SP_KEY_POINT* pt_left  = (DLCV_SP_KEY_POINT*)left;
    DLCV_SP_KEY_POINT* pt_right = (DLCV_SP_KEY_POINT*)right;
    return pt_left->reserved > pt_right->reserved ? 1 : -1;
}
*/

static void GenerateKptsData(DLCV_SP_KEY_POINT* kpts, int kpts_count, int width, int height, int upsample_scale)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }
    for(i = 0 ; i < kpts_count ; i ++)
    {
        kpts[i].x     = i % width;
        kpts[i].y     = i / width;
        kpts[i].x     = kpts[i].x * upsample_scale + (rand() % upsample_scale);
        kpts[i].y     = kpts[i].y * upsample_scale + (rand() % upsample_scale);
        kpts[i].score = ((float)(rand() % 256)) / 255.0f;
    }
/*
    for(i = 0 ; i < kpts_count ; i ++)
        kpts[i].reserved = kpts[i].y * (width * upsample_scale) + kpts[i].x;
    qsort(kpts, kpts_count, sizeof(DLCV_SP_KEY_POINT), cmp_xy);
*/
}

typedef struct tag_test_point{
    unsigned short int    x;
    unsigned short int    y;
    float                 score;
}TEST_POINT;

static const int kCmpValue[2] = { -1, 1 };
static int cmp_score_0(const void* left, const void* right)
{
    float left_val  = ((float*)left)[1];
    float right_val = ((float*)right)[1];
    return kCmpValue[left_val < right_val];
    //return left_val < right_val;
}

static int  nms_naive(TEST_POINT const*   pt_sorted,
                      int                 pt_cnt,
                      unsigned char*      nms_map,
                      int                 raw_width,
                      int                 raw_height,
                      int                 pad,
                      int                 border_size,
                      DLCV_SP_KEY_POINT*  output_kpts,
                      int                 output_kpts_count)
{
    int   i = 0, j = 0, k = 0;
    int   map_w        = raw_width  + 2 * pad;
    int   map_h        = raw_height + 2 * pad;
    int   clean_size   = 2 * pad + 1;
    int   valid_pt_cnt = 0;
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        int x = pt_sorted[i].x + pad;
        int y = pt_sorted[i].y + pad;
        nms_map[y * map_w + x] = 1;
    }

    for(i = 0 ; i < pt_cnt ; i ++)
    {
        int   x       = pt_sorted[i].x + pad;
        int   y       = pt_sorted[i].y + pad;
        int   map_pos = y * map_w + x;
        if(1 == nms_map[map_pos])
        {
            unsigned int  clean_y = pt_sorted[i].y ;
            for(j = 0 ; j < clean_size ; j ++ )
            {
                unsigned int  clean_x = pt_sorted[i].x ;
                for(k = 0 ; k < clean_size ; k ++)
                {
                    unsigned int  clean_pos = clean_y * map_w + clean_x;
                    nms_map[clean_pos] = 0;
                    clean_x ++;
                }
                clean_y ++;
            }
            nms_map[map_pos] = 255;
        }
    }

    int   valid_pt_cnt_2 = 0;
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        int   x            = pt_sorted[i].x + pad;
        int   y            = pt_sorted[i].y + pad;
        int   cur_mask_pos = y * map_w + x;
        int   cmp_res      = (255 == nms_map[cur_mask_pos]);
        if(1 == cmp_res)
        {
            valid_pt_cnt_2   = valid_pt_cnt_2 + 1;
        }
    }

    for(i = 0 ; i < border_size ; i ++)
    {
        memset(nms_map + (pad + i) * map_w, 0, map_w);
        memset(nms_map + (pad + raw_height - border_size + i) * map_w, 0, map_w);
    }
    for(i = pad + border_size ; i < (map_h - pad - border_size); i ++)
    {
        for(j = 0 ; j < border_size ; j ++)
        {
            nms_map[i * map_w + pad + j]                           = 0;
            nms_map[i * map_w + pad + raw_width - border_size + j] = 0;
        }
    }

    
    for(i = 0 ; i < pt_cnt ; i ++)
    {
        int   x            = pt_sorted[i].x + pad;
        int   y            = pt_sorted[i].y + pad;
        int   cur_mask_pos = y * map_w + x;
        int   cmp_res      = (255 == nms_map[cur_mask_pos]) & (valid_pt_cnt < output_kpts_count);
        if(1 == cmp_res)
        {
            output_kpts[valid_pt_cnt].x     = pt_sorted[i].x;
            output_kpts[valid_pt_cnt].y     = pt_sorted[i].y;
            output_kpts[valid_pt_cnt].score = pt_sorted[i].score;
            valid_pt_cnt ++;
        }
    }
    return valid_pt_cnt;
}

static int  Extract_Keypoint_Naive(const float*         prob,
                                   int                  width,
                                   int                  height,
                                   int                  channel,
                                   int                  upsample_scale,
                                   float                threshold,
                                   int                  bkg_at_last,
                                   DLCV_SP_KEY_POINT*   output_kpts,
                                   int                  output_kpts_count)
{
    TEST_POINT*  test_point = new TEST_POINT[width * upsample_scale * height * upsample_scale];
    int i = 0, j = 0, k = 0;
    int valid_point_cnt  = 0;
    float const*      cur_prob_ptr  = prob;
    int               channel_class = channel;
    if(0 != bkg_at_last)
        channel_class = channel_class - 1;
    for(i = 0 ; i < height ; i ++)
    {
        for(j = 0 ; j < width ; j ++)
        {
            for(k = 0 ; k < channel_class ; k ++)
            {
                if(cur_prob_ptr[k] >= threshold)
                {
                    test_point[valid_point_cnt].x     = j * upsample_scale + (k % upsample_scale) ;
                    test_point[valid_point_cnt].y     = i * upsample_scale + (k / upsample_scale) ;
                    test_point[valid_point_cnt].score = cur_prob_ptr[k];
                    valid_point_cnt ++;
                }
            }
            cur_prob_ptr += channel;
        }
    }

    std::qsort(test_point, valid_point_cnt, sizeof(TEST_POINT), cmp_score_0);
    int             nms_map_buf_len = (width * upsample_scale + 2 * upsample_scale) * (height * upsample_scale + 2 * upsample_scale);
    unsigned char*  nms_map         = new unsigned char[nms_map_buf_len];
    memset(nms_map, 0, nms_map_buf_len);

    valid_point_cnt = nms_naive(test_point,
                                valid_point_cnt,
                                nms_map,
                                width * upsample_scale,
                                height * upsample_scale,
                                upsample_scale,
                                upsample_scale >> 1,
                                output_kpts,
                                output_kpts_count);

    delete[] test_point;
    delete[] nms_map;
    return valid_point_cnt;
}

static void Normalize_Descriptor_Naive(const float*    feature_map,
                                       int             width,
                                       int             height,
                                       int             channel,
                                       float*          normalize_feature_map)
{
    int i = 0, j = 0;
    int img_size = width * height;
    float const*  src_ptr = feature_map;
    float*        dst_ptr = normalize_feature_map;
    for(i = 0 ; i < img_size ; i ++)
    {
        float sum     = 0.0f;
        float sum_inv = 0.0f;
        for(j = 0 ; j < channel ; j ++)
        {
            sum     += (*src_ptr) * (*src_ptr);
            src_ptr += 1;
        }
        sum     = sqrtf(sum);
        sum_inv = 1.0f / sum;
        src_ptr = src_ptr - channel;
        for(j = 0 ; j < channel ; j ++)
        {
            sum      = (*src_ptr) * sum_inv;
            *dst_ptr = sum;
            src_ptr += 1;
            dst_ptr += 1;
        }
    }
}

TEST(SuperPointPostProc, keypoint) {
    srand(time(0));

    int     res            = -1;
    int     i              = 0;
    int     src_width      = 80;
    int     src_height     = 60;
    int     channel        = 65;
    float   threshold      = 0.9;
    int     upsample_scale = 8;
    int     kpts_max_cnt   = 80*60;
    int     kpts_cnt       = 0;
    int     kpts_cnt_naive = 0;
    int     bkg_at_last    = rand() & 1;
    int     assist_buf_len = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);
    cv::Mat raw_image      = cv::imread(std::string("sample.ppm"));
    

    float*              prob         = new float[src_width * src_height * channel];
    DLCV_SP_KEY_POINT*  kpts         = new DLCV_SP_KEY_POINT[kpts_max_cnt];
    DLCV_SP_KEY_POINT*  kpts_naive   = new DLCV_SP_KEY_POINT[kpts_max_cnt];
    unsigned char*      assist_buf   = new unsigned char[assist_buf_len];
    FILE*               prob_file    = fopen("superpoint_logits_softmax_0_mnn_gpu_fp32.result_raw", "rb");
    bool                prob_file_ok = false;
    if(NULL != prob_file)
    {
        int read_res = fread(prob, src_width * src_height * channel * sizeof(float), 1, prob_file);
        (void)read_res;
        fflush(prob_file);
        fclose(prob_file);
        threshold      = 0.01;
        bkg_at_last    = 1;
        prob_file_ok   = true;
    }
    else
        GenerateFloatData(prob, src_width * src_height * channel);

    memset(kpts,         0, kpts_max_cnt * sizeof(DLCV_SP_KEY_POINT));
    memset(kpts_naive,   0, kpts_max_cnt * sizeof(DLCV_SP_KEY_POINT));
    memset(assist_buf,   0, assist_buf_len);

    printf("src: %d, %d\n", src_width, src_height);

    kpts_cnt_naive = Extract_Keypoint_Naive(prob, 
                                            src_width, 
                                            src_height, 
                                            channel, 
                                            upsample_scale, 
                                            threshold, 
                                            bkg_at_last, 
                                            kpts_naive, 
                                            kpts_max_cnt);

    kpts_cnt       = dlcv_super_point_extract_key_points(prob, 
                                                         src_width, 
                                                         src_height, 
                                                         channel, 
                                                         upsample_scale, 
                                                         threshold, 
                                                         bkg_at_last, 
                                                         assist_buf,
                                                         kpts, 
                                                         kpts_max_cnt);

    EXPECT_EQ(kpts_cnt_naive, kpts_cnt);

    printf("select point count %d\n", kpts_cnt);

    res = 0;
    for(i = 0 ; i < kpts_cnt ; i ++)
    {
        if(kpts[i].x != kpts_naive[i].x ||
           kpts[i].y != kpts_naive[i].y ||
           kpts[i].score != kpts_naive[i].score)
        {
            printf("error, %d, naive(%d, %d, %.6f), dlcv(%d, %d, %.6f)\n", 
                   i, 
                   kpts_naive[i].x,
                   kpts_naive[i].y,
                   kpts_naive[i].score,
                   kpts[i].x,
                   kpts[i].y,
                   kpts[i].score);
            res ++;
            break;
        }
    }

    EXPECT_EQ(res, 0);

    if(0 == res)
    {
        if(true == prob_file_ok && false == raw_image.empty())
        {
            cv::Mat out_image;
            cv::resize(raw_image, out_image, cv::Size(3 * src_width * upsample_scale, 3 * src_height * upsample_scale), 0.0, 0.0, cv::INTER_LINEAR);
            for(i = 0 ; i < kpts_cnt ; i ++)
                cv::circle(out_image, cv::Point(3 * kpts[i].x, 3 * kpts[i].y), 3, cv::Scalar(0, 255, 0), -1);
            cv::imwrite(std::string("superpoint_keypoint.bmp"), out_image);
        }
    }

    delete[] prob;
    delete[] kpts;
    delete[] kpts_naive;
    delete[] assist_buf;
}

TEST(SuperPointPostProc, normalize_descriptor_map) {
    srand(time(0));  

    int     res            = -1;
    int     i              = 0;
    int     src_width      = rand() % 256;
    int     src_height     = rand() % 256;
    int     channel        = rand() % 256;
    if(src_width < 32)
        src_width = 32;
    if(src_height < 32)
        src_height = 32;
    if(channel < 32)
        channel = 32;
    float*  feature_map                  = new float[src_width * src_height * channel];
    float*  normalize_feature_map        = new float[src_width * src_height * channel];
    float*  normalize_feature_map_naive  = new float[src_width * src_height * channel];

    GenerateFloatData(feature_map, src_width * src_height * channel);

    memset(normalize_feature_map,         0, src_width * src_height * channel * sizeof(float));
    memset(normalize_feature_map_naive,   0, src_width * src_height * channel * sizeof(float));

    printf("src: %d, %d, %d\n", src_width, src_height, channel);


    Normalize_Descriptor_Naive(feature_map, src_width, src_height, channel, normalize_feature_map_naive);
    dlcv_super_point_normalize_descriptor(feature_map, src_width, src_height, channel, normalize_feature_map);

    res = 0;
    for(i = 0 ; i < (src_width * src_height * channel) ; i ++)
    {
        if(fabsf(normalize_feature_map_naive[i] - normalize_feature_map[i]) >= 1e-6f)
        {
            printf("error, %d, %.6f, %.6f\n", 
                   i, 
                   normalize_feature_map_naive[i],
                   normalize_feature_map[i]);
            res ++;
            break;
        }
    }

    EXPECT_EQ(res, 0);

    delete[] feature_map;
    delete[] normalize_feature_map;
    delete[] normalize_feature_map_naive;
}

static void Extract_SuperPoint_Descriptor_Naive(const DLCV_SP_KEY_POINT*    key_points,
                                                int                         key_points_count,
                                                const float*                normalize_feature_map,
                                                int                         feature_width,
                                                int                         feature_height,
                                                int                         feature_channel,
                                                int                         upsample_scale,
                                                float*                      result)
{
    int i = 0, j = 0;
    int ratio = 1.0f / (float)(upsample_scale);
    float u = 0.0f, v = 0.0f;
    for(i = 0 ; i < key_points_count ; i ++)
    {
        int           src_x      = key_points[i].x;
        int           src_y      = key_points[i].y;
        int           blk_x0     = src_x / upsample_scale;
        int           blk_y0     = src_y / upsample_scale;
        int           blk_x1     = blk_x0 + 1;
        int           blk_y1     = blk_y0 + 1;
        int           x_in_blk   = src_x % upsample_scale;
        int           y_in_blk   = src_y % upsample_scale;
        float const*  d00        = 0;
        float const*  d01        = 0;
        float const*  d10        = 0;
        float const*  d11        = 0;
        float*        cur_result = result;
        float         u1v1       = 0.0f;
        float         u0v1       = 0.0f;
        float         u1v0       = 0.0f;
        float         u0v0       = 0.0f;
        float         sum        = 0.0f;
        u                        = x_in_blk * ratio;
        v                        = y_in_blk * ratio;
        blk_x1                   = (blk_x1 >= feature_width)  ? (feature_width - 1)  : blk_x1;
        blk_y1                   = (blk_y1 >= feature_height) ? (feature_height - 1) : blk_y1;
        d00                      = normalize_feature_map + blk_y0 * feature_width * feature_channel + blk_x0 * feature_channel;
        d01                      = normalize_feature_map + blk_y0 * feature_width * feature_channel + blk_x1 * feature_channel;
        d10                      = normalize_feature_map + blk_y1 * feature_width * feature_channel + blk_x0 * feature_channel;
        d11                      = normalize_feature_map + blk_y1 * feature_width * feature_channel + blk_x1 * feature_channel;
        u1v1                     = (1.0f - u) * (1.0f - v);
        u0v1                     = u * (1.0f - v);
        u1v0                     = (1.0f - u) * v;
        u0v0                     = u * v;
        for(j = 0 ; j < feature_channel ; j ++)
        {
            float  data00  = *d00;
            float  data01  = *d01;
            float  data10  = *d10;
            float  data11  = *d11;

            float  res0    = data00 * u1v1;
            float  res1    = data01 * u0v1;
            float  res2    = data10 * u1v0;
            float  res3    = data11 * u0v0;

            res0           = res0 + res1;
            res2           = res2 + res3;
            res0           = res0 + res2;
            *cur_result    = res0;
            sum           += (res0 * res0);
            d00           += 1;
            d01           += 1;
            d10           += 1;
            d11           += 1;
            cur_result    += 1;
        }
        sum        = sqrtf(sum);
        sum        = sum < 1e-6f ? 1e-6f : sum;
        sum        = 1.0f / sum;
        cur_result = result;
        for(j = 0 ; j < feature_channel ; j ++)
        {
            *cur_result  = *cur_result * sum;
            cur_result  += 1;
        }
        result                  += feature_channel;
    }
}

TEST(SuperPointPostProc, extract_descriptor) {
    srand(time(0));  

    int     res            = -1;
    int     i              = 0;
    int     src_width      = rand() % 256;
    int     src_height     = rand() % 256;
    int     channel        = rand() % 256;
    int     upsample_scale = rand() % 16;
    int     kpts_count     = 0; 
    if(src_width < 32)
        src_width = 32;
    if(src_height < 32)
        src_height = 32;
    if(channel < 32)
        channel = 32;
    if(upsample_scale < 2)
        upsample_scale = 2;
    kpts_count = src_width * src_height;

    float*              normalize_feature_map = new float[src_width * src_height * channel];
    DLCV_SP_KEY_POINT*  kpts                  = new DLCV_SP_KEY_POINT[kpts_count];
    float*              result                = new float[kpts_count * channel];
    float*              result_naive          = new float[kpts_count * channel];
    GenerateFloatData(normalize_feature_map, src_width * src_height * channel);
    GenerateKptsData(kpts, kpts_count, src_width, src_height, upsample_scale);

    memset(normalize_feature_map, 0, src_width * src_height * channel * sizeof(float));
    memset(result,                0, kpts_count * channel * sizeof(float));
    memset(result_naive,          0, kpts_count * channel * sizeof(float));
    printf("src: %d, %d, %d, %d, kpts_count %d\n", src_width, src_height, channel, upsample_scale, kpts_count);

    Extract_SuperPoint_Descriptor_Naive(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result_naive);
    dlcv_super_point_extract_point_descriptor(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);

    res = 0;
    for(i = 0 ; i < (kpts_count * channel) ; i ++)
    {
        if(fabsf(result[i] - result_naive[i]) >= 1e-6f)
        {
            printf("error, %d, %.6f, %.6f\n", 
                   i, 
                   result_naive[i],
                   result[i]);
            res ++;
            break;
        }
    }

    EXPECT_EQ(res, 0);

    delete[] kpts;
    delete[] normalize_feature_map;
    delete[] result;
    delete[] result_naive;
}

TEST(SuperPointPostProc, superpoint_all) {
    srand(time(0));

    int                 res                  = -1;
    int                 i                    = 0;
    int                 src_width            = rand() % 256;
    int                 src_height           = rand() % 256;
    float               threshold            = 0.01 * (rand() % 100);
    int                 upsample_scale       = rand() % 12;
    int                 kpts_max_cnt         = src_width*src_height;
    int                 kpts_cnt             = 0;
    int                 kpts_cnt_naive       = 0;
    int                 bkg_at_last          = rand() & 1;
    int                 channel              = upsample_scale * upsample_scale + bkg_at_last;
    int                 feature_channel      = rand() % 256;
    int                 assist_buf_len       = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);
    float*              prob                 = 0;
    float*              feature_map          = 0;
    float*              normalize_feature_map          = 0;
    float*              normalize_feature_map_naive    = 0;
    DLCV_SP_KEY_POINT*  kpts                           = 0;
    DLCV_SP_KEY_POINT*  kpts_naive                     = 0;
    float*              feature_result                 = 0;
    float*              feature_result_naive           = 0;
    unsigned char*      assist_buf                     = 0;

    cv::Mat             raw_image                = cv::imread(std::string("sample.ppm"));
    //FILE*               prob_file                = fopen("/home/shaquille/WorkSpace/orion_workspace/inference_workspace/build/Debug/x86_64-linux/dlcv/test/superpoint_logits_softmax_0_mnn_gpu_fp32.result_raw", "rb");
    //FILE*               feature_file             = fopen("/home/shaquille/WorkSpace/orion_workspace/inference_workspace/build/Debug/x86_64-linux/dlcv/test/superpoint_descriptors_raw_0_mnn_gpu_fp32.result_raw", "rb");
    FILE*               prob_file                = fopen("superpoint_logits_softmax_0_mnn_gpu_fp32.result_raw", "rb");
    FILE*               feature_file             = fopen("superpoint_descriptors_raw_0_mnn_gpu_fp32.result_raw", "rb");
    bool                prob_file_ok             = false;
    bool                feature_file_ok          = false;
    if(NULL != prob_file && NULL != feature_file)
    {
        src_width                    = 80;
        src_height                   = 60;
        channel                      = 65;
        feature_channel              = 128;
        upsample_scale               = 8;
        kpts_max_cnt                 = 80*60;
        assist_buf_len               = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);
        prob                         = new float[src_width * src_height * channel];
        feature_map                  = new float[src_width * src_height * feature_channel];
        normalize_feature_map        = new float[src_width * src_height * feature_channel];
        normalize_feature_map_naive  = new float[src_width * src_height * feature_channel];
        feature_result               = new float[kpts_max_cnt * feature_channel];
        feature_result_naive         = new float[kpts_max_cnt * feature_channel];
        kpts                         = new DLCV_SP_KEY_POINT[kpts_max_cnt];
        kpts_naive                   = new DLCV_SP_KEY_POINT[kpts_max_cnt];
        assist_buf                   = new unsigned char[assist_buf_len];
        int read_res = fread(prob, src_width * src_height * channel * sizeof(float), 1, prob_file);
        (void)read_res;
        fflush(prob_file);
        fclose(prob_file);
        read_res = fread(feature_map, src_width * src_height * feature_channel * sizeof(float), 1, feature_file);
        (void)read_res;
        fflush(feature_file);
        fclose(feature_file);
        threshold        = 0.01;
        bkg_at_last      = 1;
        prob_file_ok     = true;
        feature_file_ok  = true;
    }
    else
    {
        if(NULL != prob_file)
        {
            fflush(prob_file);
            fclose(prob_file);
        }
        if(NULL != feature_file)
        {
            fflush(feature_file);
            fclose(feature_file);
        }
        if(src_width < 16)
            src_width = 16;
        if(src_height < 16)
            src_height = 16;
        if(channel < 8)
            channel = 8;
        if(feature_channel < 16)
            feature_channel = 16;
        if(upsample_scale < 2)
            upsample_scale = 2;

        assist_buf_len               = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);
        channel                      = upsample_scale * upsample_scale + bkg_at_last;
        kpts_max_cnt                 = src_width * src_height;
        prob                         = new float[src_width * src_height * channel];
        feature_map                  = new float[src_width * src_height * feature_channel];
        normalize_feature_map        = new float[src_width * src_height * feature_channel];
        normalize_feature_map_naive  = new float[src_width * src_height * feature_channel];
        feature_result               = new float[kpts_max_cnt * feature_channel];
        feature_result_naive         = new float[kpts_max_cnt * feature_channel];
        kpts                         = new DLCV_SP_KEY_POINT[kpts_max_cnt];
        kpts_naive                   = new DLCV_SP_KEY_POINT[kpts_max_cnt];
        assist_buf                   = new unsigned char[assist_buf_len];
        GenerateFloatData(prob,        src_width * src_height * channel);
        GenerateFloatData(feature_map, src_width * src_height * feature_channel);
    }

    memset(kpts,                        0, kpts_max_cnt * sizeof(DLCV_SP_KEY_POINT));
    memset(kpts_naive,                  0, kpts_max_cnt * sizeof(DLCV_SP_KEY_POINT));
    memset(normalize_feature_map,       0, src_width * src_height * feature_channel * sizeof(float));
    memset(normalize_feature_map_naive, 0, src_width * src_height * feature_channel * sizeof(float));
    memset(feature_result,              0, kpts_max_cnt * feature_channel * sizeof(float));
    memset(feature_result_naive,        0, kpts_max_cnt * feature_channel * sizeof(float));
    memset(assist_buf,                  0, assist_buf_len);

    printf("src: w %d, h %d, c0 %d, c1 %d, s %d\n", src_width, src_height, channel, feature_channel, upsample_scale);

    kpts_cnt_naive = Extract_Keypoint_Naive(prob, 
                                            src_width, 
                                            src_height, 
                                            channel, 
                                            upsample_scale, 
                                            threshold, 
                                            bkg_at_last, 
                                            kpts_naive, 
                                            kpts_max_cnt);
    Normalize_Descriptor_Naive(feature_map, 
                               src_width, 
                               src_height, 
                               feature_channel, 
                               normalize_feature_map_naive);
    Extract_SuperPoint_Descriptor_Naive(kpts_naive, 
                                        kpts_cnt_naive, 
                                        normalize_feature_map_naive, 
                                        src_width, 
                                        src_height, 
                                        feature_channel, 
                                        upsample_scale, 
                                        feature_result_naive);
    kpts_cnt    = dlcv_super_point_extract_key_points(prob, 
                                                      src_width, 
                                                      src_height, 
                                                      channel, 
                                                      upsample_scale, 
                                                      threshold, 
                                                      bkg_at_last, 
                                                      assist_buf,
                                                      kpts, 
                                                      kpts_max_cnt);
    dlcv_super_point_normalize_descriptor(feature_map, 
                                          src_width, 
                                          src_height, 
                                          feature_channel, 
                                          normalize_feature_map);
    dlcv_super_point_extract_point_descriptor(kpts, 
                                              kpts_cnt,
                                              normalize_feature_map,
                                              src_width, 
                                              src_height, 
                                              feature_channel, 
                                              upsample_scale,
                                              feature_result);
    EXPECT_EQ(kpts_cnt_naive, kpts_cnt);

    printf("select point count %d\n", kpts_cnt);

    res = 0;
    for(i = 0 ; i < kpts_cnt ; i ++)
    {
        if(kpts[i].x != kpts_naive[i].x ||
           kpts[i].y != kpts_naive[i].y ||
           kpts[i].score != kpts_naive[i].score)
        {
            printf("keypoint error, %d, naive(%d, %d, %.6f), dlcv(%d, %d, %.6f)\n", 
                   i, 
                   kpts_naive[i].x,
                   kpts_naive[i].y,
                   kpts_naive[i].score,
                   kpts[i].x,
                   kpts[i].y,
                   kpts[i].score);
            res ++;
            break;
        }
    }
    EXPECT_EQ(res, 0);

    res = 0;
    for(i = 0 ; i < (src_width * src_height * feature_channel) ; i ++)
    {
        if(fabsf(normalize_feature_map[i] - normalize_feature_map_naive[i]) > DIFF_EPS)
        {
            printf("normalize error, %d, %.6f, %.6f\n", 
                   i, 
                   normalize_feature_map_naive[i],
                   normalize_feature_map[i]);
            res ++;
            break;
        }
    }
    EXPECT_EQ(res, 0);

    res = 0;
    for(i = 0 ; i < (kpts_cnt * feature_channel) ; i ++)
    {
        if(fabsf(feature_result[i] - feature_result_naive[i]) > DIFF_EPS)
        {
            printf("feature error, %d, %.6f, %.6f\n", 
                   i, 
                   feature_result_naive[i],
                   feature_result[i]);
            res ++;
            break;
        }
    }
    EXPECT_EQ(res, 0);

    if(0 == res)
    {
        if(true == prob_file_ok && true == feature_file_ok && false == raw_image.empty())
        {
            cv::Mat out_image;
            cv::resize(raw_image, out_image, cv::Size(3 * src_width * upsample_scale, 3 * src_height * upsample_scale), 0.0, 0.0, cv::INTER_LINEAR);
            for(i = 0 ; i < kpts_cnt ; i ++)
                cv::circle(out_image, cv::Point(3 * kpts[i].x, 3 * kpts[i].y), 3, cv::Scalar(0, 255, 0), -1);
            cv::imwrite(std::string("superpoint_keypoint.bmp"), out_image);
        }
    }

    delete[] prob;
    delete[] feature_map;
    delete[] normalize_feature_map;
    delete[] normalize_feature_map_naive;
    delete[] feature_result;
    delete[] feature_result_naive;
    delete[] kpts;
    delete[] kpts_naive;
    delete[] assist_buf;
}

TEST(SuperPointPostProc, performance_keypoint)
{
    int     i              = 0;
    int     src_width      = 80;
    int     src_height     = 60;
    int     channel        = 65;
    float   threshold      = 0.9;
    int     upsample_scale = 8;
    int     kpts_max_cnt   = 80 * 60;
    int     kpts_cnt       = 0;
    int     loop           = 1000;
    int     res            = 0;
    int     bkg_at_last    = rand() & 1;
    int     assist_buf_len = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);

    float*              prob       = new float[src_width * src_height * channel];
    DLCV_SP_KEY_POINT*  kpts       = new DLCV_SP_KEY_POINT[kpts_max_cnt];
    unsigned char*      assist_buf = new unsigned char[assist_buf_len];
    FILE*               prob_file  = fopen("superpoint_logits_softmax_0_mnn_gpu_fp32.result_raw", "rb");
    if(NULL != prob_file)
    {
        int read_res = fread(prob, src_width * src_height * channel * sizeof(float), 1, prob_file);
        (void)read_res;
        fflush(prob_file);
        fclose(prob_file);
        threshold      = 0.01;
        bkg_at_last    = 1;
    }
    else
        GenerateFloatData(prob, src_width * src_height * channel);

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    //std::qsort(0, 0, 0, 0);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        kpts_cnt = dlcv_super_point_extract_key_points(prob, 
                                                       src_width, 
                                                       src_height, 
                                                       channel, 
                                                       upsample_scale, 
                                                       threshold, 
                                                       bkg_at_last, 
                                                       assist_buf,
                                                       kpts, 
                                                       kpts_max_cnt);
        //kpts_cnt = Extract_Keypoint_Naive(prob, src_width, src_height, channel, upsample_scale, threshold, bkg_at_last, kpts, kpts_max_cnt);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("select point count %d\n", kpts_cnt);

    printf("performance_keypoints, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);

    delete[] prob;
    delete[] kpts;
    delete[] assist_buf;
} 

/*
int shuffle_idx_array(int cnt, int* hash_table, int* idx_out)
{
    int i          = 0 ;
    int iIdx       = 0 ;
    int remain_cnt = cnt ;
    int iLastValue = 0 ;
    int iTemp      = 0 ;

    if (cnt <= 0)
        return 0 ;

    if (1 == cnt)
        idx_out[0] = (rand() % cnt) ;

    for (i = 0; i < cnt ; i++)
        hash_table[i] = i ;

    for (i = 0; i < cnt ; i++)
    {
        if (remain_cnt > 1)
        {
            iIdx                        = rand() % (remain_cnt - 1);
            iLastValue                  = hash_table[remain_cnt - 1] ;
            iTemp                       = hash_table[iIdx] ;
            hash_table[iIdx]            = iLastValue ;
            hash_table[remain_cnt - 1]  = iTemp ;
            idx_out[i]                  = iTemp ;
        }
        else
            idx_out[i] = hash_table[0] ;

        remain_cnt-- ;
    }

    return 0 ;
}
*/
TEST(SuperPointPostProc, performance_qsort)
{
    srand(time(0));
    TEST_POINT*     kpts2     = 0;
    TEST_POINT*     kpts3     = 0;
    int             i         = 0;
    int             point_cnt = 0;
    //FILE*           save_file = fopen("/home/shaquille/WorkSpace/orion_workspace/inference_workspace/build/Debug/x86_64-linux/dlcv/test/unsort_point.raw", "rb");
    FILE*           save_file = fopen("unsort_point.raw", "rb");
    if(NULL != save_file)
    {
        fseek(save_file, 0L, SEEK_END);
        int length = ftell(save_file);
        fseek(save_file, 0L, SEEK_SET);
        point_cnt = length / sizeof(TEST_POINT);
        kpts2     = new TEST_POINT[point_cnt];
        kpts3     = new TEST_POINT[point_cnt];
        int read_res = fread(kpts2, 1, point_cnt * sizeof(TEST_POINT), save_file);
        (void)read_res;
        fflush(save_file);
        fclose(save_file);
        printf("%d point read\n", point_cnt);
        memcpy(kpts3, kpts2, point_cnt * sizeof(TEST_POINT));
    }

    int                      loop      = 1000;
    TEST_POINT*              kpts0     = new TEST_POINT[point_cnt];
    TEST_POINT*              kpts1     = new TEST_POINT[point_cnt];
    for(i = 0 ; i < point_cnt ; i ++)
    {
        kpts0[i].score = 0.9f + 0.1f * (((float)(rand() % 256)) / 255.0f);
    }
    memcpy(kpts1, kpts0, point_cnt * sizeof(TEST_POINT));

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    //std::qsort(0, 0, 0, 0);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
    {
        std::qsort(kpts0, point_cnt, sizeof(TEST_POINT), cmp_score_0);
        memcpy(kpts0, kpts1, point_cnt * sizeof(TEST_POINT));
    }
        
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);
    printf("performance_cmp_score_0, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
    {
        std::qsort(kpts2, point_cnt, sizeof(TEST_POINT), cmp_score_0);
        memcpy(kpts2, kpts3, point_cnt * sizeof(TEST_POINT));
    }
        
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);
    printf("performance_cmp_score_1, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    delete[] kpts0;
    delete[] kpts1;
    delete[] kpts2;
    delete[] kpts3;
} 

TEST(SuperPointPostProc, performance_normalize_descriptor_map)
{
    int     i              = 0;
    int     src_width      = 80;
    int     src_height     = 60;
    int     channel        = 128;
    int     loop           = 1000;
    int     res            = 0;
    float*  feature_map                  = new float[src_width * src_height * channel];
    float*  normalize_feature_map        = new float[src_width * src_height * channel];

    GenerateFloatData(feature_map, src_width * src_height * channel);

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        dlcv_super_point_normalize_descriptor(feature_map, src_width, src_height, channel, normalize_feature_map);
        //Normalize_Feature_Naive(feature_map, src_width, src_height, channel, normalize_feature_map);
        //dlcv_super_point_extract_point_feature(0, 0, feature_map, src_width, src_height, channel, normalize_feature_map, 0, 0);
        
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_normalize_descriptor_map, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);

    delete[] feature_map;
    delete[] normalize_feature_map;
} 

TEST(SuperPointPostProc, performance_extract_descriptor)
{
    int     i              = 0;
    int     src_width      = 80;
    int     src_height     = 60;
    int     channel        = 128;
    int     upsample_scale = 8;
    int     kpts_count     = 60 * 80; 
    int     loop           = 1000;
    int     res            = 0;

    float*              normalize_feature_map = new float[src_width * src_height * channel];
    DLCV_SP_KEY_POINT*  kpts                  = new DLCV_SP_KEY_POINT[kpts_count];
    float*              result                = new float[kpts_count * channel];
    GenerateFloatData(normalize_feature_map, src_width * src_height * channel);
    GenerateKptsData(kpts, kpts_count, src_width, src_height, upsample_scale);

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
        dlcv_super_point_extract_point_descriptor(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //Extract_SuperPoint_Feature_Naive(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //dlcv_super_point_extract_point_feature(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //Extract_SuperPoint_Feature_Naive(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //dlcv_super_point_extract_point_feature(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //Extract_SuperPoint_Feature_Naive(kpts, kpts_count, normalize_feature_map, src_width, src_height, channel, upsample_scale, result);
        //dlcv_super_point_extract_point_feature(0, 0, feature_map, src_width, src_height, channel, normalize_feature_map, 0, 0);
        
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("performance_extract_descriptor, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);

    delete[] kpts;
    delete[] normalize_feature_map;
    delete[] result;
} 

TEST(SuperPointPostProc, performance_superpoint_all)
{
    int                 i                        = 0;
    int                 src_width                = 80;
    int                 src_height               = 60;
    int                 channel                  = 65;
    int                 feature_channel          = 128;
    float               threshold                = 0.9;
    int                 upsample_scale           = 8;
    int                 kpts_max_cnt             = 80 * 60;
    int                 kpts_cnt                 = 0;
    int                 loop                     = 1000;
    int                 res                      = 0;
    int                 bkg_at_last              = 1;
    int                 assist_buf_len           = 3 * (src_width * upsample_scale + 2 * upsample_scale) * (src_height * upsample_scale + 2 * upsample_scale) * sizeof(float);
    float*              prob                     = new float[src_width * src_height * channel];
    DLCV_SP_KEY_POINT*  kpts                     = new DLCV_SP_KEY_POINT[kpts_max_cnt];
    float*              feature_map              = new float[src_width * src_height * feature_channel];
    float*              normalize_feature_map    = new float[src_width * src_height * feature_channel];
    float*              feature_result           = new float[kpts_max_cnt * feature_channel];
    unsigned char*      assist_buf               = new unsigned char[assist_buf_len];
    FILE*               prob_file                = fopen("superpoint_logits_softmax_0_mnn_gpu_fp32.result_raw", "rb");
    FILE*               feature_file             = fopen("superpoint_descriptors_raw_0_mnn_gpu_fp32.result_raw", "rb");
    bool                prob_file_ok             = false;
    bool                feature_file_ok          = false;
    if(NULL != prob_file && NULL != feature_file)
    {
        int read_res = fread(prob, src_width * src_height * channel * sizeof(float), 1, prob_file);
        (void)read_res;
        fflush(prob_file);
        fclose(prob_file);
        read_res = fread(feature_map, src_width * src_height * feature_channel * sizeof(float), 1, feature_file);
        (void)read_res;
        fflush(feature_file);
        fclose(feature_file);
        threshold        = 0.01;
        bkg_at_last      = 1;
        prob_file_ok     = true;
        feature_file_ok  = true;
    }
    else
        GenerateFloatData(prob, src_width * src_height * channel);

	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost;

    //std::qsort(0, 0, 0, 0);
    tm_start = std::chrono::system_clock::now();
    for(i = 0; i < loop ; i ++)
    {
        kpts_cnt    = dlcv_super_point_extract_key_points(prob, 
                                                        src_width, 
                                                        src_height, 
                                                        channel, 
                                                        upsample_scale, 
                                                        threshold, 
                                                        bkg_at_last, 
                                                        assist_buf,
                                                        kpts, 
                                                        kpts_max_cnt);
        dlcv_super_point_normalize_descriptor(feature_map, 
                                              src_width, 
                                              src_height, 
                                              feature_channel, 
                                              normalize_feature_map);
        dlcv_super_point_extract_point_descriptor(kpts, 
                                                  kpts_cnt,
                                                  normalize_feature_map,
                                                  src_width, 
                                                  src_height, 
                                                  feature_channel, 
                                                  upsample_scale,
                                                  feature_result);
    }
        //kpts_cnt = Extract_Keypoint_Naive(prob, src_width, src_height, channel, upsample_scale, threshold, bkg_at_last, kpts, kpts_max_cnt);
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);

    printf("select point count %d\n", kpts_cnt);

    printf("performance_superpoint_all, %d rounds cost: %lldus, avg: %.6fus\r\n", 
           loop,
           (long long int)(tm_cost.count()), 
           ((double)tm_cost.count()) / ((double)loop));

    EXPECT_EQ(res, 0);

    delete[] prob;
    delete[] kpts;
    delete[] feature_map;
    delete[] normalize_feature_map;
    delete[] feature_result;
    delete[] assist_buf;
} 
