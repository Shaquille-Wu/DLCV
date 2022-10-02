#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator> 
#include "util.h"
#include "logging.h"
#include "dlcv.h"

using namespace std;
namespace vision{

std::string g_json_path;

std::string get_dlcv_version_string()
{
    stringstream ss;
    ss<<DLCV_VERSION_MAJOR<<"."<<DLCV_VERSION_MINOR<<"."<<DLCV_VERSION_TINY;
    return ss.str();
}

void cc_print_mat(const string tag, cv::Mat& mat){
    double min, max;
    cv::minMaxLoc(mat, &min, &max);
    cv::Scalar s = cv::sum(mat);
    int h = mat.rows;
    int w = mat.cols;
    int c = mat.channels();
    double sum = cv::sum(s)[0];
    double mean= sum/(w*h*c);
    char buf[256] = {0};
    sprintf(buf, "%16s (%d, %d, %d) sum mean min max: %.5f %.5f %.5f %.5f\n",
        tag.c_str(), h, w, c, sum, mean, (float)min, (float)max);
    LOG(INFO) << buf;
}

void cc_print_dim(const string tag, Dim<float>* ptensor){
    if (ptensor == nullptr or ptensor->empty()) {
        char buf[256] = {0};
        sprintf(buf, "%16s empty\n", tag.c_str());
        LOG(ERROR) << buf;
        return;
    }

    vector<int> dims = ptensor->_dims;
    std::stringstream shape;
    std::copy(dims.begin(), dims.end(), std::ostream_iterator<int>(shape, ", "));

    float* pp = ptensor->_ptr;
    double sum = 0;
    float v, min = 10000, max = -10000;
    for (int i = 0; i < ptensor->_size; i++){
        v = *(pp++);
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    float mean = sum / (ptensor->_size);
    char buf[256] = {0};
    sprintf(buf, "%16s (%s) sum mean min max: %.5f %.5f %.5f %.5f\n",
        tag.c_str(), shape.str().c_str(), sum, mean, min, max);
    LOG(INFO) << buf;
}

string get_file_path(const char* file_name) {
    string path;
#if defined(__ANDROID__)
    string file_path(file_name);
    auto rpos = file_path.rfind('/');
    if(rpos != std::string::npos) {
        path = file_path.substr(0, rpos+1);
    }
    else {
        path = string("./");
    }
    return path;
#elif defined(_WIN32)
	string file_path(file_name);
	auto rpos = file_path.rfind('\\');
	if (rpos != std::string::npos) {
		path = file_path.substr(0, rpos + 1);
	}
	else {
		path = string(".\\");
	}
	return path;
#else
    char abs_path_buff[PATH_MAX];
    if(realpath(file_name, abs_path_buff)){
        string abs_file(abs_path_buff);
        path = abs_file.substr(0, abs_file.rfind('/')+1);
    }
    return path;
#endif
}
}

