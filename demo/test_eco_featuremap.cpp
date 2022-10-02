#include <iostream>
#include "opencv2/opencv.hpp"
#include <getopt.h>
#include <sys/time.h>
#include "featuremap_runner.h"          // for FeaturemapRunner, FeaturemapRunner::output...
#include "logging.h"
#include "timer.h"

using namespace std;
using namespace vision;

using vision::FeaturemapRunner;
using vision::ElementType;

int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cout << argv[0] << " cfgfile imgfile" << std::endl;
        return -1;
    }

    std::string cfgfile = argv[1];
    std::string imgfile = argv[2];
    cv::Mat image = cv::imread(imgfile);

    auto featuremap_runner = new FeaturemapRunner(cfgfile);
    auto output = FeaturemapRunner::output_type();
    HighClock clk;
    clk.Start();
    featuremap_runner->run(image, output);
    clk.Stop();
    std::cout << "featuremap_runner run time:" << clk.GetTime() / 1000 << " ms." << std::endl;

    auto& feature_map = output.front();
    const auto channel = feature_map.shape[1];
    const auto height = feature_map.shape[2];
    const auto width = feature_map.shape[3];
    int datasize = channel * height * width;

    if (feature_map.data.expired()) {
        LOG(ERROR) << "feature_map.data is expired";
        return 1;
    }

    float* ptr = reinterpret_cast<float *>(feature_map.data.lock().get());

    LOG(INFO) << "Eco Feature result:";
    std::string outdata = "\n";
    for(int i=0;i<datasize;++i) {
        outdata += std::to_string(ptr[i]) + " ";
        if((i+1) % 16 == 0) {
            outdata += "\n";
        }
    }
    LOG(INFO) << outdata;

    return 0;
}
