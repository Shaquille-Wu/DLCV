#ifndef _INFERENCE_PROXY_H_
#define _INFERENCE_PROXY_H_

#include "IInference.h"
#include "json.hpp"
#include "blob.h"

#ifndef  _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

using namespace std;

namespace vision{

//IInference* CreateInference(const char* so_path);
//IInference* CreateInferenceWithConfig(const nlohmann::json& engine);

class Inference {
    IInference* _pinf;

public:
    Inference(const nlohmann::json& config);
    int run();
    ~Inference();
    inline const std::vector<std::string>& input_names() const { return _inname; }
    inline const std::vector<std::string>& output_names() const { return _outname; }

    vector<Blob> _indata;
    vector<Blob> _outdata;
    bool _debug;
    vector<std::string> _outname;
    vector<std::string> _inname;
};

}
#endif
