
#include<fstream>
#include "modelrunner.h"
#include "logging.h"
#include "error_code.h"
#include "util.h"
#include <chrono>

using namespace std;
using namespace nlohmann; // json

namespace vision{    

static const int kPreprocessMax = 8;

json load_json(const string config_file){
    json cfg;
    try {
        std::ifstream inJsonFile(config_file.c_str());
        if (!inJsonFile) {
            LOG(ERROR) << " Error: json file open failed: " << config_file;
            ABORT();
        }
        inJsonFile >> cfg;
    }
    catch (std::exception& e) {
        LOG(ERROR) << " Error: json file read failed: " << e.what();
        ABORT();
    }
    return cfg;
}

ModelRunner::ModelRunner(const string cfgfile, const string jpatch){
    g_json_path = get_file_path(cfgfile.c_str());
    json cfg = load_json(cfgfile);
    if(!jpatch.empty()){
        json json_patch = json::parse(jpatch);
        cfg = cfg.patch(json_patch);
        LOG(INFO) << "cfg patched: " << json_patch << endl;
    }
    LOG(INFO) << "cfg: " << cfg << endl;

    _inf = new Inference(cfg.at("inference"));
    if(true == cfg.contains("preprocess"))
    {
        if(true == cfg.at("preprocess").is_array())
        {
            std::vector<nlohmann::json> json_vec       = cfg.at("preprocess");
            int                         valid_json_cnt = kPreprocessMax >= ((int)(json_vec.size())) ? ((int)(json_vec.size())) : kPreprocessMax;
            if(kPreprocessMax < ((int)(json_vec.size())))
            {
                LOG(INFO) << " Error: preprocess count in json beyond the max: " << kPreprocessMax;
            }
            _preprocessor.resize(json_vec.size(), nullptr);
            for(int i = 0 ; i < (int)(json_vec.size()) ; i ++)
                _preprocessor[i]  = new Preprocessor(json_vec[i]);
        }
        else
        {
            _preprocessor.resize(1, nullptr);
            _preprocessor[0] = new Preprocessor(cfg.at("preprocess"));
        }
    }
    else
    {
        LOG(ERROR) << " Error: json file read failed, cannot find preprocess";
        ABORT();
    }

    _postprocessor = new Postprocessor(cfg.at("postprocess"));

}

int ModelRunner::run(cv::Mat& img, vision::DLCVOut& out){
    // reset result
    out.reset();

    PreprocInfo pre_proc_info;
    int prep_ret = _preprocessor[0]->run(img, pre_proc_info, _inf->_indata[0]);
    if (prep_ret != 0)
        return prep_ret;

    int inf_ret = _inf->run();
    if (inf_ret != 0)
        return inf_ret;

    int post_ret = _postprocessor->run(_inf->_outdata, _inf->output_names(), pre_proc_info, out);
    if (post_ret != 0)
        return post_ret;

    //LOG(INFO) << "performance cost: " << (long long int)(tm_cost0.count()) << " "  << (long long int)(tm_cost1.count()) << " " << (long long int)(tm_cost2.count());

    return 0;
}

int ModelRunner::run_images(std::vector<cv::Mat*>& imgs, vision::DLCVOut& out)
{
    // reset result
    out.reset();

    PreprocInfo pre_proc_info[kPreprocessMax] ;
    for(auto i = 0 ; i < _preprocessor.size() ; i ++)
    {
        int prep_ret = _preprocessor[i]->run(*(imgs[i]), pre_proc_info[i], _inf->_indata[i]);
        if (prep_ret != 0)
            return prep_ret;
    }

    int inf_ret = _inf->run();
    if (inf_ret != 0)
        return inf_ret;

    int post_ret = _postprocessor->run(_inf->_outdata, _inf->output_names(), pre_proc_info[0], out);
    if (post_ret != 0)
        return post_ret;

    return 0;
}

ModelRunner::~ModelRunner(){
    for(auto i = 0 ; i < _preprocessor.size() ; i ++)
    {
        if (_preprocessor[i]){
            delete _preprocessor[i];
            _preprocessor[i] = NULL;
        }
    }

    if (_inf){
        delete _inf;
        _inf = NULL;
    }
    if (_postprocessor){
        delete _postprocessor;
        _postprocessor = NULL;
    }
}

}
