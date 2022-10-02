#ifndef _UTIL_H_
#define _UTIL_H_

#include <numeric>
#include <string>
#include <iterator>

#include "dim.h"
#include "blob.h"
#include "opencv2/opencv.hpp"

#undef min
#undef max

namespace vision{

extern std::string g_json_path;

std::string get_file_path(const char* file_name);

static inline std::string get_abs_path(std::string& file) {
    if(file[0] == '/') {
        return file;
    }

    return g_json_path + file;
}

void cc_print_mat(const std::string tag, cv::Mat& mat);
void cc_print_dim(const std::string tag, vision::Dim<float>* ptensor);

template<typename T>
void cc_print_blob(const std::string tag, const Blob &blob){
    if (blob.empty()) {
        char buf[256] = {0};
        sprintf(buf, "%16s empty\n", tag.c_str());
        LOG(ERROR) << buf;
        return; 
    }

    std::stringstream shape;
    const auto& dims = blob.size();
    std::copy(dims.begin(), dims.end(), std::ostream_iterator<int>(shape, ", "));

    double sum = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    T* raw = blob.get<T>();
    for (auto i = 0; i < blob.num_elements(); ++i) {
        double value = raw[i];
        sum += value;
        min = std::min(value, min);
        max = std::max(value, max);
    }
    auto mean = sum / blob.num_elements();
    char buf[256] = {0};
    sprintf(buf, "%16s (%s) sum mean min max: %.5f %.5f %.5f %.5f\n",
        tag.c_str(), shape.str().c_str(), sum, mean, min, max);
    LOG(INFO) << buf;
}


}

#endif
