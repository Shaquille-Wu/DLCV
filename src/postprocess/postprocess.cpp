
#include "postprocess.h"
#include "det_ssd_post.h"
#include "featuremap_post.h"
#include "multiclass_post.h"
#include "reidfeature_post.h"
#include "kps_attribute_post.h"
#include "logging.h"

using namespace std;
using json=nlohmann::json;

namespace vision{

// PostprocOp
PostprocOp::PostprocOp() {
    _debug = false;
}

int PostprocOp::run(const vector<Blob> &indata, const std::vector<std::string>& names, const PreprocInfo& info, vision::DLCVOut& out) {
  return run(indata, info, out);
}

// Register all PostprocOp
typedef std::map<std::string, PostprocOp*(*)(const json& param)> PostprocMap;
PostprocMap postproc_map{
    {"det_ssd_post", &make_postproc<DetSSDPost>},
    {"det_ssd_detectionout_post", &make_postproc<DetSSDDetectionoutPost>},
    {"featuremap_post", &make_postproc<FeaturemapPost>},
    {"multiclass_post", &make_postproc<MultiClassPost>},
    {"reidfeature_post", &make_postproc<ReidFeaturePost>},
    {"kps_attribute_post", &make_postproc<KeypointsAttributePost>},
};

// Postprocessor
Postprocessor::Postprocessor(const json& cfg){
    LOG(INFO) << "Postprocessor init";

    _debug = cfg.at("debug");
    vector<json> ps = cfg.at("ops");     
    for(unsigned int i = 0; i < ps.size(); i++){
        string type = ps[i].at("type");
        json param  = ps[i].at("param");
        // validation of op type
        if (postproc_map.find(type) == postproc_map.end()){
            LOG(ERROR) << "Error: postproc op type not supported: " << type << ". Supported postproc ops are: ";
            for(PostprocMap::iterator it = postproc_map.begin(); it != postproc_map.end(); ++it) {
              LOG(ERROR) << "    " << it->first;
            }
            ABORT();
        }
        // create op
        PostprocOp* op = postproc_map[type](param);
        op->_debug = _debug;
        _postproc_ops.push_back(op);
    } 
}

int Postprocessor::run(const vector<Blob>& indata, const vector<string>& names, const PreprocInfo& info, vision::DLCVOut& out){
    for (unsigned int i = 0; i < _postproc_ops.size(); i++){
        int r = _postproc_ops[i]->run(indata, names, info, out);
        if (r != 0)
            return r;
    }
    return 0; 
}

Postprocessor::~Postprocessor(){
    for (unsigned int i = 0;  i < _postproc_ops.size(); i++){
        if (_postproc_ops[i] != NULL){
            delete _postproc_ops[i];
            _postproc_ops[i] = NULL;
        }
    }
}

}
