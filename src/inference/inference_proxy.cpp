#include "inference_proxy.h"
#include "error_code.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>

#ifndef  _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include "util.h"
#include "logging.h"

using namespace std;
using json=nlohmann::json;

namespace vision{

IInference* CreateInference(const char* so_path)
{
  IInference* inferPtr = NULL;
  typedef void*  (*fpCreateInference)();
#ifndef  _WIN32 // ANDROID, Linux
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle){
        fprintf(stderr, "[%s](%d) dlopen get error: %s\n", __FILE__, __LINE__, dlerror());
        exit(EXIT_FAILURE);
    }

    do {
        fpCreateInference pCreateInference = (fpCreateInference)dlsym(handle, "CreateInference");
        inferPtr = (IInference*)(*pCreateInference)();
    } while (0); // for resource handle
    //dlclose(handle); // Do NOT close it, because other models may still be using this so
#else  
    HINSTANCE hInstLibrary = LoadLibraryA(so_path);
    if (hInstLibrary != NULL){
        fpCreateInference pCreateInference = (fpCreateInference)GetProcAddress(hInstLibrary, "CreateInference");
        if (pCreateInference != NULL)
        {
          inferPtr = (IInference*)(*pCreateInference)();
        }
    }
#endif
    return inferPtr;
}

IInference* CreateInferenceWithConfig(const nlohmann::json& cfg) {

    std::string engine_path = cfg.at("engine");
    IInference* ptr =  vision::CreateInference(engine_path.c_str());
    CHECK(ptr != NULL);
    const char* engine_ver = ptr->INF_version();
    LOG(INFO) << "engine: " << engine_path.c_str() << ", version: " << engine_ver;

    json param  = cfg.at("engine_param");
    for (nlohmann::json::const_iterator it = param.begin(); it != param.end(); ++it) {
        bool  type_valid = true;
        vision::EngineFeature   new_feature;
        strncpy(new_feature.keyName, it.key().c_str(), 64);
        if(it.value().is_number_float())
        {
            new_feature.value.fValue = it.value();
            new_feature.valueType = ValueType::FLOAT;
        }
        else if (it.value().is_number()) {
            new_feature.value.nValue = int(it.value());
            new_feature.valueType = ValueType::INT;
        }
        else if (it.value().is_string()) {
            new_feature.valueType = ValueType::BYTES;
            std::string value = it.value();
            strncpy((char*)new_feature.value.ucValue, value.c_str(), 256);
        }
        else {
            LOG(INFO) << (char*)new_feature.keyName << " type undef";
            type_valid = false;
        }

        if (type_valid) {
            ptr->INF_set_engine_feature(&new_feature);
        }
    }
    return ptr;

}

// interface class for inference
Inference::Inference(const json& config) {
    LOG(INFO) << "Inference init";
    _debug = config.at("debug");
    _pinf =  CreateInferenceWithConfig(config);

    // create indata buffer
    vector<json> incfg = config.at("inputs");
    for (unsigned int i = 0 ; i < incfg.size(); i++){
        string name       = incfg[i].at("name");
        _inname.push_back(name);
        const std::vector<int> shape = incfg[i].at("shape");
        CHECK(shape.size() == 4);

        // insert i-th input blob into _indata & get pointer to allocated memory
        // note: default value of "inputs[i].dtype" is "float32"
        void* data_ptr = nullptr;
        if (incfg[i].contains("dtype") and incfg[i].at("dtype") == "int8") {
            // case: INT8
            _indata.push_back(Blob::create<uint8_t>(shape));
            data_ptr = _indata.back().get<uint8_t>();
        } else {
            // case: FLOAT32
            _indata.push_back(Blob::create<float>(shape));
            data_ptr = _indata.back().get<float>();
        }
        CHECK(data_ptr != nullptr);
        
        int r = _pinf->INF_set_data(name.c_str(), data_ptr, shape, INPUT);
        if (r != ERROR_CODE_SUCCESS) {
            LOG(ERROR) << "Error: INF_set_data failed, return code " <<  r;
            ABORT();
        }
    }

    // create outdata buffer
    vector<json> outcfg = config.at("outputs");
    for (unsigned int i = 0 ; i < outcfg.size(); i++){
        string name       = outcfg[i].at("name");
        _outname.push_back(name);
        const std::vector<int> shape = outcfg[i].at("shape");
        CHECK(shape.size() == 4);

        // insert i-th output blob into _outdata & get pointer to allocated memory
        // note: default value of "outputs[i].dtype" is "float32"
        void* data_ptr = nullptr;
        if (outcfg[i].contains("dtype") and outcfg[i].at("dtype") == "int8") {
            // case: INT8
            _outdata.push_back(Blob::create<uint8_t>(shape));
            data_ptr = _outdata.back().get<uint8_t>();
        } else {
            // case: FLOAT32
            _outdata.push_back(Blob::create<float>(shape));
            data_ptr = _outdata.back().get<float>();
        }
        CHECK(data_ptr != nullptr);

        int r = _pinf->INF_set_data(name.c_str(), data_ptr, shape, OUTPUT);
        if (r != ERROR_CODE_SUCCESS) {
            LOG(ERROR) << "Error: INF_set_data failed, return code " << r;
            ABORT();
        }
    }

    // Load model
    std::string model_path = config.at("model");
    model_path = get_abs_path(model_path);
    if (_pinf->INF_load_model(model_path.c_str(), 0) != 0) {
        LOG(ERROR) << "Error: load model failed";
        ABORT();
    }
}

int Inference::run() {
    if (_debug) {
        for (auto i = 0; i < _indata.size(); ++i) {
            const Blob& blob = _indata[i];
            if (blob.contains<float>()) {
              cc_print_blob<float>("indata[float32]", blob);
            } else if (blob.contains<uint8_t>()) {
              cc_print_blob<uint8_t>("indata[int8]", blob);
            }
        }
    }

    int r = _pinf->INF_forward(1); //batchsize = 1

    if (_debug) {
        for (auto i = 0; i < _outdata.size(); ++i) {
            const Blob& blob = _outdata[i];
            if (blob.contains<float>()) {
              cc_print_blob<float>("outdata[float32]", blob);
            } else if (blob.contains<uint8_t>()) {
              cc_print_blob<uint8_t>("outdata[int8]", blob);
            }
        }
    }

    if (r != ERROR_CODE_SUCCESS) {
        LOG(ERROR) << "Error: inference failed, return code " << r;
        return r;
    }
    return ERROR_CODE_SUCCESS;
}

Inference::~Inference(){
    if (_pinf){
        delete _pinf;
        _pinf = NULL; 
    }
}

}
