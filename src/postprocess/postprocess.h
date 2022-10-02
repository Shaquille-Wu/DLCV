#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <type_traits>
#include "json.hpp"
#include "dim.h"
#include "dlcv.h"
#include "preprocess.h"

using namespace std;
using json=nlohmann::json; //json

namespace vision{

// PostprocOp
class PostprocOp {
public:
    PostprocOp();
    virtual ~PostprocOp(){};

    /**
     * @brief This variant of `run` is the legacy routine without name lookup support.
     * It's pure virtual for maximum compatibility: by now, most post-processors DO NOT need name lookup support.
     * 
     * @param indata all \c Blob configured in inference.outputs
     * @param info \c PreprocInfo to reconstruct structual results, detection boxes for example
     * @param out \c DLCVOut to contain all processed results
     */
    virtual int run(const vector<Blob> &indata, const PreprocInfo& info, vision::DLCVOut& out) = 0;

    /**
     * @brief This variant of `run` support name lookup for post processors.
     * 
     * The default implementation just calls above \c run function.
     * 
     * @param indata all \c Blob configured in inference.outputs
     * @param names corresponding names of \c indata, i.e., names[i] = inference.outputs[i].name 
     * @param info \c PreprocInfo to reconstruct structual results, detection boxes for example
     * @param out \c DLCVOut to contain all processed results
     */
    virtual int run(const vector<Blob> &indata, const vector<string>& names, const PreprocInfo& info, vision::DLCVOut& out);
    bool _debug;
};

// function to make ops 
template<typename T, typename=typename std::is_base_of<PostprocOp, T>::type> 
PostprocOp* make_postproc(const json& param) { 
    return new T(param); 
};

// Postprocessor
class Postprocessor {
    vector<PostprocOp* > _postproc_ops;
    bool _debug;
public:
    Postprocessor(const json& cfg);
    ~Postprocessor();

    /**
     * @brief Sequentially run all post processors.
     * 
     * @see PostprocOp::run
     * @param indata all \c Blob configured in inference.outputs
     * @param names corresponding names of \c indata, i.e., names[i] = inference.outputs[i].name 
     * @param info \c PreprocInfo to reconstruct structual results, detection boxes for example
     * @param out \c DLCVOut to contain all processed results
     */
    int run(const vector<Blob>& indata, const vector<string>& names, const PreprocInfo& info, vision::DLCVOut& out);
};


}
#endif
