#include "reidfeature_post.h"
#include "error_code.h"
#include "logging.h"

namespace vision {

ReidFeaturePost::ReidFeaturePost(const json &node) {
}

int ReidFeaturePost::run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut &out) {
    CHECK (indata.size() == 1);
    out.has_feature = true;
    
    const auto& blob = indata.front(); 
    const float* pd = blob.get<float>();
    int size = blob.num_elements();

    out.feature.resize(size);
    memcpy(&out.feature[0], pd, size * blob.element_size());
    return 0;
}

} // namespace vision

