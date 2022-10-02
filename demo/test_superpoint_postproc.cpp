#include <memory>                        // for shared_ptr
#include <string>                        // for string
#include <tuple>                         // for tie, ignore, tuple
#include <vector>                        // for vector
#include <opencv2/opencv.hpp>
#include <chrono>
#include "featuremap_runner.h"
#include "../dlcv_proc_opt/dlcv_proc_opt.h"
#include "logging.h"

static void nchw2nhwc(const float* nchw, int h, int w, int c, float* nhwc)
{
    int i = 0, j = 0, k = 0;
    for(i = 0 ; i < c ; i ++)
    {
        for(j = 0 ; j < h ; j ++)
        {
            for(k = 0 ; k < w ; k ++)
            {
                int   src_pos  = i * h * w + j * w + k;
                int   dst_pos  = j * w * c + k * c + i;
                *(nhwc + dst_pos) = *(nchw + src_pos);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        return 1;
    }

    std::string   cfgfile      = argv[1];
    std::string   imgfile      = argv[2];
    bool          is_prob_nchw = false;
    if(argc >= 4)
        is_prob_nchw = atoi(argv[3]);
    cv::Mat       src_image  = cv::imread(imgfile);
    if(true == src_image.empty())
    {
        LOG(ERROR) << "cannot read image from: " << imgfile;
        return 0;
    }

    auto featuremap_runner   = new vision::FeaturemapRunner(cfgfile);
    auto output              = vision::FeaturemapRunner::output_type();
    featuremap_runner->run(src_image, output);

    int                 loop                   = 100;
    int                 i                      = 0;
    auto&               prob_map               = output[0];
    auto&               desc_map               = output[1];
    const int           h                      = desc_map.shape[1];  //it should be 60
    const int           w                      = desc_map.shape[2];  //it should be 80
    const int           prob_c                 = false == is_prob_nchw ? prob_map.shape[3] : prob_map.shape[1];  //it should be 65
    const int           desc_c                 = desc_map.shape[3];  //it should be 128
    const int           upsample               = 8;
    const float         threshold              = 0.01f;
    const int           kpt_max_cnt            = w * h;
    //const int           normalize_desc_buf_len = w * h * desc_c * sizeof(float);
    int                 assist_buf_len         = 3 * (w * upsample + 2 * upsample) * (w * upsample + 2 * upsample) * sizeof(float);
    //assist_buf_len = assist_buf_len < normalize_desc_buf_len ? normalize_desc_buf_len : assist_buf_len;
    unsigned char*      assist_buf             = new unsigned char[assist_buf_len];
    DLCV_SP_KEY_POINT*  kpts                   = new DLCV_SP_KEY_POINT[kpt_max_cnt];
    float*              kpts_desc              = new float[kpt_max_cnt * desc_c];
    float*              prob_data_nhwc         = false == is_prob_nchw ? 0 : new float[w * h * prob_c];
    int                 key_pt_cnt             = 0;
	std::chrono::time_point<std::chrono::system_clock> tm_start;
	std::chrono::time_point<std::chrono::system_clock> tm_end;
	std::chrono::microseconds                          tm_cost, tm_cost_post;
    long long int       post_proc_cost_sum     = 0;

    tm_start = std::chrono::system_clock::now();
    for(i = 0 ; i < loop ; i ++)
    {
        featuremap_runner->run(src_image, output);

        float*  prob_data  = (float*)(output[0].data.lock().get());
        float*  desc_data  = (float*)(output[1].data.lock().get());
        if(true == is_prob_nchw)
        {
            nchw2nhwc(prob_data, h, w, prob_c, prob_data_nhwc);
            prob_data = prob_data_nhwc;
        }
	    std::chrono::time_point<std::chrono::system_clock> tm_start_post;
	    std::chrono::time_point<std::chrono::system_clock> tm_end_post;
        std::chrono::microseconds                          tm_cost_post;
        tm_start_post      = std::chrono::system_clock::now();
        key_pt_cnt         = dlcv_super_point_extract_key_points(prob_data,
                                                                 w,
                                                                 h,
                                                                 prob_c,
                                                                 upsample,
                                                                 threshold,
                                                                 1,
                                                                 assist_buf,
                                                                 kpts,
                                                                 kpt_max_cnt);
        if(key_pt_cnt > 0)
        {
            //we think it will avoid cache miss if we set output as input
            //so, we can speed up our "normalize"
            //otherwise, we should alloc new buf for normalize_desc
            dlcv_super_point_normalize_descriptor(desc_data, w, h, desc_c, desc_data);
            dlcv_super_point_extract_point_descriptor(kpts, 
                                                      key_pt_cnt, 
                                                      desc_data, 
                                                      w, 
                                                      h, 
                                                      desc_c, 
                                                      upsample, 
                                                      kpts_desc);
            //kpts_descriptor will be applied into feature_match
            //our demo will be ignore it
        }
        tm_end_post         = std::chrono::system_clock::now();
        tm_cost_post        = std::chrono::duration_cast<std::chrono::microseconds>(tm_end_post - tm_start_post);
        post_proc_cost_sum += (long long int)(tm_cost_post.count());
    }
    tm_end   = std::chrono::system_clock::now();
    tm_cost  = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start);
    LOG(INFO) << "found " << key_pt_cnt << " point(s)";
    LOG(INFO) << "performance " << loop << " rounds cost: " << (long long int)(tm_cost.count()) << "us, avg: " << (((double)tm_cost.count()) / ((double)loop)) << "us " << "postproc: " << (((double)post_proc_cost_sum) / ((double)loop));

    cv::Mat out_image;
    cv::resize(src_image, out_image, cv::Size(3 * w * upsample, 3 * h * upsample), 0.0, 0.0, cv::INTER_LINEAR);
    for(i = 0 ; i < key_pt_cnt ; i ++)
        cv::circle(out_image, cv::Point(3 * kpts[i].x, 3 * kpts[i].y), 3, cv::Scalar(0, 255, 0), -1);
    cv::imwrite(std::string("superpoint_keypoint.bmp"), out_image);

    delete[] assist_buf;
    delete[] kpts;
    delete[] kpts_desc;
    if(NULL != prob_data_nhwc)
        delete prob_data_nhwc;
    prob_data_nhwc = NULL;
    return 0;
}

