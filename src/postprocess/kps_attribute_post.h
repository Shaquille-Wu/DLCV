#ifndef _KPS_ATTRIBUTE_POST_H_
#define _KPS_ATTRIBUTE_POST_H_ 

#include "postprocess.h"

using namespace std;
using json = nlohmann::json;

namespace vision {

class KeypointsAttributePost : public PostprocOp {
    bool do_reorder_;
    std::string keypoints_type_; // heat_map or regress
    std::string data_format_;    // nchw or nhwc, for heat_map only
    std::vector<int> kps_idx_range_; // size=2, incuding ending, like [0,211] 
    std::vector<std::vector<int> > orders_;
    bool have_multiclass_;

public:
    KeypointsAttributePost(const json& param);
    int run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut& out);

};

}  // namespace vision

#endif  //_KPS_ATTRIBUTE_POST_H_
