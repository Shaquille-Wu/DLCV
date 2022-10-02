
#include<iostream>
#include "det_ssd_post.h"
#include "error_code.h"
#include "logging.h"

using namespace std;
using json=nlohmann::json;

namespace vision{

vector<Box>& insert_map(map<string, vector<Box> >& d, string key, Box& new_val) {
    map<string, vector<Box> >::iterator it = d.find(key);
    if (it != d.end()) {
        it->second.push_back(new_val);
    }
    else {
        vector<Box> v;
        v.push_back(new_val);
        d.insert(make_pair(key, v));
        return d[key];
    }
    return it->second;
}

// det_ssd_post
DetSSDPost::DetSSDPost(const json& param) {
    conf_thresh_ = param.at("conf_thresh");
    json lm = param.at("label_map");
    for(json::const_iterator it = lm.begin(); it != lm.end(); ++it){
        label_map_.insert({int(it.value()), it.key()});
    }
}
int DetSSDPost::run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut& out){
    CHECK (indata.size() == 1);
    
    // get preproc_info
    int origin_h = info.origin_h;
    int origin_w = info.origin_w; 

    map<string, vector<Box> > boxes_map;
    const auto& blob = indata.front();
    const float* pd = blob.get<float>();
    int num    = blob.size(2); // shape (1,1,100,7)
    int stride = blob.size(3); // shape (1,1,100,7)
    int shift = stride - 6;
    for (int i = 0; i < num; i++) {
        Box b;
        b.cls = int(pd[i * stride + shift + 0]);
        b.score = pd[i * stride + shift + 1];
        if (b.score < conf_thresh_)
            continue;
        b.x1 = std::max(pd[i * stride + shift + 2] * origin_w, 0.f);
        b.y1 = std::max(pd[i * stride + shift + 3] * origin_h, 0.f);
        b.x2 = std::min(pd[i * stride + shift + 4] * origin_w, (float)origin_w);
        b.y2 = std::min(pd[i * stride + shift + 5] * origin_h, (float)origin_h);
        if(b.x2 <= b.x1 || b.y2 <= b.y1) {
            continue;
        }
        insert_map(boxes_map, label_map_[b.cls], b);
        //LOG(INFO) << i << "/" << num << " " << b.cls << " " << b.score << " " << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2;
    }
    // set result
    out.has_boxes = true;
    out.boxes_map = boxes_map;

    return 0;
}

DetSSDDetectionoutPost::DetSSDDetectionoutPost(const json& param) : conf_thresh_(0.0f), ptr_detection_out_impl_(nullptr)
{
    conf_thresh_ = param.at("conf_thresh");
    json lm = param.at("label_map");
    for(json::const_iterator it = lm.begin(); it != lm.end(); ++it){
        label_map_.insert({int(it.value()), it.key()});
    }    
    std::unique_ptr<DetectionOutImpl>  ptr_detection_out(new DetectionOutImpl);
    int                                res = ptr_detection_out->Read(param);
    if(0 != res)   return;
    res = ptr_detection_out->Setup();
    if(0 != res)   return;

    ptr_detection_out_impl_ = std::move(ptr_detection_out);
}

int DetSSDDetectionoutPost::run(const vector<Blob> &indata, const PreprocInfo& info, vision::DLCVOut& out)
{
    CHECK (indata.size() >= 2);
    CHECK (nullptr != ptr_detection_out_impl_);

    // get preproc_info
    int      src_img_width   = info.origin_w;
    int      src_img_height  = info.origin_h;

    const auto& conf = indata[0];
    const auto& loc = indata[1];
    const float*   conf_data       = conf.get<float>();
    const float*   loc_data        = loc.get<float>();
    int      conf_data_count = conf.size(2) * conf.size(3);  // dim: { 1, 1, xxxx, 2 }
    int      loc_data_count  = loc.size(2) * loc.size(3);  // dim: { 1, 1, xxxx, 4 }
    int      res             = ptr_detection_out_impl_->Solve(conf_data, 
                                                              conf_data_count, 
                                                              loc_data, 
                                                              loc_data_count);
    if(0 == res)
    {
        const float* result_data       = ptr_detection_out_impl_->GetResultData();
        int          result_item_count = ptr_detection_out_impl_->GetResultItemCount();
        int          result_item_dim   = DetectionOutLayer::kResultItemElementCount;
        int          i                 = 0;
        std::map<std::string, vector<Box> > boxes_map;
        for(i = 0 ; i < result_item_count ; i ++)
        {
            Box b;
            b.cls   = (int)(result_data[i * result_item_dim + 1] + 0.5f);
            b.score = result_data[i * result_item_dim + 2];
            if (b.score < conf_thresh_)
                continue;
            b.x1 = std::max(result_data[i * result_item_dim + 3] * src_img_width, 0.f);
            b.y1 = std::max(result_data[i * result_item_dim + 4] * src_img_height, 0.f);
            b.x2 = std::min(result_data[i * result_item_dim + 5] * src_img_width, (float)src_img_width);
            b.y2 = std::min(result_data[i * result_item_dim + 6] * src_img_height, (float)src_img_height);
            if(b.x2 <= b.x1 || b.y2 <= b.y1) {
                continue;
            }
            insert_map(boxes_map, label_map_[b.cls], b);
        }
        // set result
        out.has_boxes = true;
        out.boxes_map = boxes_map;
    }
    else
    {
        res = ERROR_CODE_PARAM_SIZE;
    }
    
    return res;
}

}
