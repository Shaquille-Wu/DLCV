#ifndef _DET_SSD_POST_H_
#define _DET_SSD_POST_H_

#include "postprocess.h"
#include "./detection_out/detection_out_impl.h"


using namespace std;
using json=nlohmann::json;

namespace vision{

// det_ssd_post without DetectionOutput
class DetSSDPost : public PostprocOp {
    float conf_thresh_;
    map<int, string> label_map_;
public:
    DetSSDPost(const json& param);
    int run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut& out);
};

// det_ssd_detectionout_post WITH DetectionOutput
class DetSSDDetectionoutPost : public PostprocOp {
public:
    DetSSDDetectionoutPost(const json& param);
    int run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut& out);

private:
    float                                   conf_thresh_;
    std::map<int, std::string>              label_map_;
    std::unique_ptr<DetectionOutImpl>       ptr_detection_out_impl_;
};

}
#endif
