#include "kps_attribute_post.h"

#include <iostream>

#include "error_code.h"
#include "logging.h"

using namespace std;
using json = nlohmann::json;

namespace vision {

static void keypoints_reorder(vector<cv::Point>& keypoints, const vector<vector<int> >& orders) {
    // size check
    unsigned int size_sum = 0;
    for (int i=0; i < orders.size(); i++){
        size_sum += orders[i].size();
    }
    CHECK(size_sum == keypoints.size());

    float* outx = new float[size_sum];
    float* outy = new float[size_sum];
    int start = 0;
    for (unsigned int i = 0; i < orders.size(); i++){
        const vector<int>& order = orders[i];
        for (int j = 0; j < (int)order.size(); j++) {
            int k = order[j];
            outx[start+j] = keypoints[k].x;
            outy[start+j] = keypoints[k].y;
        }
        start += (int)order.size();
    }
    // set output
    for (unsigned int i = 0; i < size_sum; i++) {
        keypoints[i].x = outx[i];
        keypoints[i].y = outy[i];
    }
    delete[] outx;
    delete[] outy;
}

KeypointsAttributePost::KeypointsAttributePost(const json& param) {
    // 1. keypoints_type

    CHECK(param.contains("keypoints_type"));
    {
        keypoints_type_ = param.at("keypoints_type");
        CHECK(keypoints_type_ == "heat_map" or keypoints_type_ == "regress"); // only regress or heat_map
        // 2. data_format(optional)
        if (keypoints_type_ == "heat_map") {
            data_format_ = param.at("data_format"); //for heat_map, data_format was needed
            CHECK(data_format_ == "nchw" or data_format_ == "nhwc");
        }
    }

    CHECK(param.contains("keypoints_idx_range"));
    {
        // 3. keypoints_idx_range
        param.at("keypoints_idx_range").get_to(kps_idx_range_); //including ending [0, 211]
        CHECK(kps_idx_range_.size() == 2);
    }

    // 4. keypoints_orders
    int sum = 0;
    if (param.contains("keypoints_orders")) {
        vector<json> ps = param.at("keypoints_orders"); //must have this to show output order    
        do_reorder_ = false;
        for (unsigned int i = 0; i < ps.size(); i++) {
            std::string name = ps[i].at("name");
            vector<int> out_idx_range = ps[i].at("out_idx_range"); //including ending, like [0,32], [33, 52] 
            vector<int> order = ps[i].at("reorder");
            if (order.size() > 0 or do_reorder_) {
                do_reorder_ = true;
                CHECK(order.size() == out_idx_range[1] - out_idx_range[0] + 1);
            }
            sum += out_idx_range[1] - out_idx_range[0] + 1;
            orders_.push_back(order);
        }
    }

    have_multiclass_ = true;
    if(param.find("have_multiclass") != param.end()) {
        have_multiclass_ = param.at("have_multiclass").get<bool>();
    }

    // check num_kps
    if (keypoints_type_ == "heat_map"){
        CHECK(sum == kps_idx_range_[1] - kps_idx_range_[0] + 1);
    }
    else{ //regress
        CHECK(sum == int((kps_idx_range_[1] - kps_idx_range_[0] + 1) / 2));
    }
}
int KeypointsAttributePost::run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut& out) {
    CHECK(indata.size() >= 1);
    // get preproc_info
    int origin_h = info.origin_h;
    int origin_w = info.origin_w;
    int attri_idx = 0;
    
    // process keypoints in indata[0]. we assume indata[0] contains keypoints
    if (keypoints_type_ == "heat_map"){ // heat_map
        auto& blob = indata[0];
        CHECK(kps_idx_range_[1] < blob.size(1));
        const int height = blob.size(2);
        const int width = blob.size(3);
        const float* blob_ptr = blob.get<float>();
        for(int i=kps_idx_range_[0]; i < kps_idx_range_[1]+1; ++i) { //loop on channels
            CHECK(data_format_ == "nchw"); //TODO support nhwc
            const float* hm = blob_ptr + (i * width * height);

            float max_val = hm[0];
            float max_idx_h = 0;
            float max_idx_w = 0;
            for(int h = 0; h < height; ++h) {
                for(int w = 0; w < width; ++w) {
                    if(hm[h * width + w] > max_val) {
                        max_val = hm[h * width + w];
                        max_idx_h = static_cast<float>(h);
                        max_idx_w = static_cast<float>(w);
                    }
                }
            }
            // finetune point
            if(max_idx_h > 0 && max_idx_h < (height - 1) && max_idx_w > 0 && max_idx_w < (width - 1)) {
                const float top = hm[((int)max_idx_h - 1) * width + (int)max_idx_w];
                const float bottom = hm[((int)max_idx_h + 1) * width + (int)max_idx_w];
                const float left = hm[(int)max_idx_h * width + (int)max_idx_w - 1];
                const float right = hm[(int)max_idx_h * width + (int)max_idx_w + 1];
                if(bottom > top) {
                    max_idx_h += 0.25;
                } else if(bottom < top) {
                    max_idx_h -= 0.25;
                }
                if(right > left) {
                    max_idx_w += 0.25;
                } else if(right < left) {
                    max_idx_w -= 0.25;
                }
            }
            cv::Point p( (max_idx_w * origin_w) / width , (max_idx_h * origin_h) / height );
            out.keypoints.push_back(p);
        }
        out.has_keypoints = true;
    }
    else { // keypoints_type_ == regress, just copy result from indata 
        int num_kps = int((kps_idx_range_[1]-kps_idx_range_[0] + 1) / 2 );
        const float* pd = indata[0].get<float>();   
        size_t len =  indata[0].num_elements(); 
        CHECK( kps_idx_range_[1] < len);

        int kps_idx = kps_idx_range_[0];
        for (int i = 0; i < num_kps; i++) {
            cv::Point p;
            p.x = pd[kps_idx + i * 2] * origin_w;
            p.y = pd[kps_idx + i * 2 + 1] * origin_h;
            out.keypoints.push_back(p);
        }
        out.has_keypoints = true;

        if (kps_idx_range_[0] != 0 || kps_idx_range_[1]  != len-1){ // has attribute in indata[0]
            out.multiclass.resize(indata.size());
            for (int i = 0; i < kps_idx_range_[0]; i++) {
                out.multiclass[0].push_back(pd[i]);
            }
            for (int i = kps_idx_range_[1] + 1; i < len; i++) {
                out.multiclass[0].push_back(pd[i]);
            }
            attri_idx = 1;
            out.has_multiclass = true;
        }
        else{
            out.multiclass.resize(indata.size()-1);
            attri_idx = 0;
        }
    }
    if(do_reorder_){
        keypoints_reorder(out.keypoints, orders_);
    }
    
    // process other attributes
    if(have_multiclass_) {
        if ( indata.size() > 1)
            out.has_multiclass = true;
        for(size_t i = 1; i < indata.size(); ++i) {
            const auto& blob = indata[i];
            const float* pd = blob.get<float>();
            int size = blob.num_elements();
            out.multiclass[attri_idx+i-1].resize(size);
            memcpy(&(out.multiclass[attri_idx+i-1][0]), pd, size * blob.element_size());
        }
    }

    return 0;
}
}  // namespace vision
