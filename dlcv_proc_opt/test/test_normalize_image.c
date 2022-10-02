#include "../dlcv_proc_opt.h"
#include "test_com_def.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void  normalize_image_uc3f3_naive(const unsigned char*  src,
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

void  normalize_image_uc4f4_naive(const unsigned char*  src,
                                  float*                dst,
                                  int                   width,
                                  int                   height,
                                  int                   src_line_size,
                                  int                   dst_line_size,
                                  const float*          mean,
                                  const float*          std)
{
    float inv_std[4]   = { 1.0f / std[0], 1.0f / std[1], 1.0f / std[2], 1.0f / std[3] };
    int   line_ele_cnt = 4 * width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 4)
        {
            dst[i * dst_line_size + j]     = ((float)(src[i * src_line_size + j])     - mean[0]) * inv_std[0];
            dst[i * dst_line_size + j + 1] = ((float)(src[i * src_line_size + j + 1]) - mean[1]) * inv_std[1];
            dst[i * dst_line_size + j + 2] = ((float)(src[i * src_line_size + j + 2]) - mean[2]) * inv_std[2];
            dst[i * dst_line_size + j + 3] = ((float)(src[i * src_line_size + j + 3]) - mean[3]) * inv_std[3];
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

void  normalize_image_f3f3_naive(const float*  src,
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

void  normalize_image_f4f4_naive(const float*  src,
                                 float*        dst,
                                 int           width,
                                 int           height,
                                 int           src_line_size,
                                 int           dst_line_size,
                                 const float*  mean,
                                 const float*  std)
{
    float inv_std[4]   = { 1.0f / std[0], 1.0f / std[1], 1.0f / std[2], 1.0f / std[3] };
    int   line_ele_cnt = 4 * width;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < line_ele_cnt; j += 4)
        {
            dst[i * dst_line_size + j]     = (src[i * src_line_size + j]     - mean[0]) * inv_std[0];
            dst[i * dst_line_size + j + 1] = (src[i * src_line_size + j + 1] - mean[1]) * inv_std[1];
            dst[i * dst_line_size + j + 2] = (src[i * src_line_size + j + 2] - mean[2]) * inv_std[2];
			dst[i * dst_line_size + j + 3] = (src[i * src_line_size + j + 3] - mean[3]) * inv_std[3];
        }
    }
}

static int test_normalize_uc1f1()
{
    int             res       = -1;
    int             i         = 0;
    int             width     = 1 + (rand() % 2048);
    int             height    = 1 + (rand() % 2048);
    int             channel   = 1;
    unsigned char*  src       = (unsigned char*)malloc(width * height * channel * sizeof(unsigned char));
    float*          dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*          dst       = (float*)malloc(width * height * channel * sizeof(float));
    float           mean[]    = {
        106.0f
    };
    float   std[] = {
        51.1f
    };

    memset(src,       0, width * height * channel * sizeof(unsigned char));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_uc1f1, width %d, height %d\n", width, height);

    GenerateUCData(src, width * height * channel);

    normalize_image_uc1f1_naive(src, dst_naive, width, height, width, width, mean[0], std[0]);
    dlcv_normalize_image_uc1f1(src, dst, width, height, width, width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
            goto test_normalize_uc1f1_leave;
        }
    }

    res = 0;
test_normalize_uc1f1_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_uc3f3()
{
    int             res       = -1;
    int             i         = 0;
    int             width     = (1 + (rand() % 2048));
    int             height    = 1 + (rand() % 2048);
    int             channel   = 3;
    unsigned char*  src       = (unsigned char*)malloc(width * height * channel * sizeof(unsigned char));
    float*          dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*          dst       = (float*)malloc(width * height * channel * sizeof(float));
    float           mean[]    = {
        106.0f,
        127.1f,
        133.3f,
    };
    float   std[] = {
        51.1f,
        54.5f,
        56.7f,
    };

    memset(src,       0, width * height * channel * sizeof(unsigned char));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_uc3f3, width %d, height %d\n", width, height);

    GenerateUCData(src, width * height * channel);

    normalize_image_uc3f3_naive(src, dst_naive, width, height, 3 * width, 3 * width, mean, std);
    dlcv_normalize_image_uc3f3(src, dst, width, height, 3 * width, 3 * width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
            goto test_normalize_ucf3_leave;
        }
    }

    
    res = 0;
test_normalize_ucf3_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_uc4f4()
{
    int             res       = -1;
    int             i         = 0;
    int             width     = (1 + (rand() % 2048));
    int             height    = 1 + (rand() % 2048);
    int             channel   = 4;
    unsigned char*  src       = (unsigned char*)malloc(width * height * channel * sizeof(unsigned char));
    float*          dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*          dst       = (float*)malloc(width * height * channel * sizeof(float));
    float           mean[]    = {
        106.0f,
        127.1f,
        133.3f,
		145.5f,
    };
    float   std[] = {
        51.1f,
        54.5f,
        56.7f,
		58.9f,
    };

    memset(src,       0, width * height * channel * sizeof(unsigned char));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_uc4f4, width %d, height %d\n", width, height);

    GenerateUCData(src, width * height * channel);

    normalize_image_uc4f4_naive(src, dst_naive, width, height, 4 * width, 4 * width, mean, std);
    dlcv_normalize_image_uc4f4(src, dst, width, height, 4 * width, 4 * width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
		{
			printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
			goto test_normalize_uc4f4_leave;
		}
            
    }

    res = 0;
test_normalize_uc4f4_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_f1f1()
{
    int     res       = -1;
    int     i         = 0;
    int     width     = 1 + (rand() % 2048);
    int     height    = 1 + (rand() % 2048);
    int     channel   = 1;
    float*  src       = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst       = (float*)malloc(width * height * channel * sizeof(float));
    float   mean[]    = {
        106.0f
    };
    float   std[] = {
        51.1f
    };

    memset(src,       0, width * height * channel * sizeof(float));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_f1f1, width %d, height %d\n", width, height);

    GenerateFloatData(src, width * height * channel);

    normalize_image_f1f1_naive(src, dst_naive, width, height, width, width, mean[0], std[0]);
    dlcv_normalize_image_f1f1(src, dst, width, height, width, width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
            goto test_normalize_f1_leave;
        }
    }

    res = 0;
test_normalize_f1_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_f3f3()
{
    int     res       = -1;
    int     i         = 0;
    int     width     = (1 + (rand() % 2048));
    int     height    = 1 + (rand() % 2048);
    int     channel   = 3;
    float*  src       = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst       = (float*)malloc(width * height * channel * sizeof(float));
    float   mean[]    = {
        106.0f,
        127.1f,
        133.3f,
    };
    float   std[] = {
        51.1f,
        54.5f,
        56.7f,
    };

    memset(src,       0, width * height * channel * sizeof(float));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_f3f3, width %d, height %d\n", width, height);

    GenerateFloatData(src, width * height * channel);

    normalize_image_f3f3_naive(src, dst_naive, width, height, 3 * width, 3 * width, mean, std);
    dlcv_normalize_image_f3f3(src, dst, width, height, 3 * width, 3 * width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
        {
            printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
            goto test_normalize_f3_leave;
        }
    }

    
    res = 0;
test_normalize_f3_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_f4f4()
{
    int     res       = -1;
    int     i         = 0;
    int     width     = (1 + (rand() % 2048));
    int     height    = 1 + (rand() % 2048);
    int     channel   = 4;
    float*  src       = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst       = (float*)malloc(width * height * channel * sizeof(float));
    float   mean[]    = {
        106.0f,
        127.1f,
        133.3f,
		145.5f,
    };
    float   std[] = {
        51.1f,
        54.5f,
        56.7f,
		58.9f,
    };

    memset(src,       0, width * height * channel * sizeof(float));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));

    printf("normalize_f4f4, width %d, height %d\n", width, height);

    GenerateFloatData(src, width * height * channel);

    normalize_image_f4f4_naive(src, dst_naive, width, height, 4 * width, 4 * width, mean, std);
    dlcv_normalize_image_f4f4(src, dst, width, height, 4 * width, 4 * width, mean, std);

    for(i = 0 ; i < (width * height * channel) ; i ++)
    {
        if(dst_naive[i] != dst[i])
		{
			printf("error, %d, %.6f, %.6f\n", i, dst_naive[i], dst[i]);
			goto test_normalize_f4_leave;
		}
            
    }

    res = 0;
test_normalize_f4_leave:
    free(src);
    free(dst_naive);
    free(dst);

    return res ;
}

static int test_normalize_performance()
{
    int     loop      = 10000;
    int     res       = -1;
    int     i         = 0;
    int     width     = 640;
    int     height    = 480;
    int     channel   = 4;
    float*  src       = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst_naive = (float*)malloc(width * height * channel * sizeof(float));
    float*  dst       = (float*)malloc(width * height * channel * sizeof(float));
    float   mean[]    = {
        106.0f,
        127.1f,
        133.3f,
		145.5f,
    };
    float   std[] = {
        51.1f,
        54.5f,
        56.7f,
		58.9f,
    };
    unsigned long long   f1_cost = 0;
    unsigned long long   f3_cost = 0;
    unsigned long long   f4_cost = 0;

    memset(src,       0, width * height * channel * sizeof(float));
    memset(dst,       0, width * height * channel * sizeof(float));
    memset(dst_naive, 0, width * height * channel * sizeof(float));
    printf("test_normalize_performance, width %d, height %d\n", width, height);
    GenerateFloatData(src, width * height * channel);

    printf("dlcv_normalize_image_uc1f1 start\n");
    f1_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_uc1f1((const unsigned char*)src, dst, width, height, width, width, mean, std);
    f1_cost = GetOrionTimeUs() - f1_cost;
    printf("dlcv_normalize_image_uc1f1 cost: %lldus, avg: %.6fus\n", f1_cost, ((double)f1_cost)/((double)loop));

    printf("dlcv_normalize_image_uc3f3 start\n");
    f3_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_uc3f3((const unsigned char*)src, dst, width, height, 3 * width, 3 * width, mean, std);
    f3_cost = GetOrionTimeUs() - f3_cost;
    printf("dlcv_normalize_image_uc3f3 cost: %lldus, avg: %.6fus\n", f3_cost, ((double)f3_cost)/((double)loop));

    printf("dlcv_normalize_image_uc4f4 start\n");
    f4_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_uc4f4((const unsigned char*)src, dst, width, height, 4 * width, 4 * width, mean, std);
    f4_cost = GetOrionTimeUs() - f4_cost;
    printf("dlcv_normalize_image_uc4f4 cost: %lldus, avg: %.6fus\n", f4_cost, ((double)f4_cost)/((double)loop));

    printf("dlcv_normalize_image_f1f1 start\n");
    f1_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_f1f1(src, dst, width, height, width, width, mean, std);
    f1_cost = GetOrionTimeUs() - f1_cost;
    printf("dlcv_normalize_image_f1f1 cost: %lldus, avg: %.6fus\n", f1_cost, ((double)f1_cost)/((double)loop));

    printf("dlcv_normalize_image_f3f3 start\n");
    f3_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_f3f3(src, dst, width, height, 3 * width, 3 * width, mean, std);
    f3_cost = GetOrionTimeUs() - f3_cost;
    printf("dlcv_normalize_image_f3f3 cost: %lldus, avg: %.6fus\n", f3_cost, ((double)f3_cost)/((double)loop));

    printf("dlcv_normalize_image_f4f4 start\n");
    f4_cost = GetOrionTimeUs();
    for(i = 0 ; i < loop ; i ++)
        dlcv_normalize_image_f4f4(src, dst, width, height, 4 * width, 4 * width, mean, std);
    f4_cost = GetOrionTimeUs() - f4_cost;
    printf("dlcv_normalize_image_f4f4 cost: %lldus, avg: %.6fus\n", f4_cost, ((double)f4_cost)/((double)loop));

    return 0;
}

int main(int argc, char** argv)
{
    int res = 0;
	srand(time(0));
    res = test_normalize_uc1f1();
    if(0 != res)
    {
        printf("test_normalize_uc1f1 is error\n");
        return res;
    }
    res = test_normalize_uc3f3();
    if(0 != res)
    {
        printf("test_normalize_uc3f3 is error\n");
        return res;
    }
    res = test_normalize_uc4f4();
    if(0 != res)
    {
        printf("test_normalize_uc4f4 is error\n");
        return res;
    }

    res = test_normalize_f1f1();
    if(0 != res)
    {
        printf("test_normalize_f1f1 is error\n");
        return res;
    }
    res = test_normalize_f3f3();
    if(0 != res)
    {
        printf("test_normalize_f3f3 is error\n");
        return res;
    }
    res = test_normalize_f4f4();
    if(0 != res)
    {
        printf("test_normalize_f4f4 is error\n");
        return res;
    }

    res = test_normalize_performance();
    if(0 != res)
    {
        printf("test_normalize_performance is error\n");
        return res;
    }

    return 0;
}