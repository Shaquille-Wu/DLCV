#include "multiclass_post.h"
#include "error_code.h"

namespace vision {

MultiClassPost::MultiClassPost(const json &node) {
}

int MultiClassPost::run(const vector<Blob>& indata, const PreprocInfo& info, vision::DLCVOut &out) {
    out.has_multiclass = true;
    out.multiclass.resize(indata.size());
    for(size_t i=0;i<indata.size();++i) {
        const auto& blob = indata[i]; 
        const float* pd = blob.get<float>();
        int size = blob.num_elements();
        out.multiclass[i].resize(size);
        memcpy(&out.multiclass[i][0], pd, size * blob.element_size());
    }
    return 0;
}

} // namespace vision

