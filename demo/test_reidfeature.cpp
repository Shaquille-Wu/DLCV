#include <iostream>
#include "reidfeature.h"
#include "opencv2/opencv.hpp"
#include <getopt.h>
#include <sys/time.h>
#include "logging.h"

using namespace std;
using namespace vision;

using vision::Feature;

int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cout << argv[0] << " cfgfile imgfile" << std::endl;
        return -1;
    }

    std::string cfgfile = argv[1];
    std::string imgfile = argv[2];

    auto output = Feature::output_type();
    auto feature = new Feature(cfgfile);
    cv::Mat image = cv::imread(imgfile);
    feature->run(image, output);

    int idx = 0;
    LOG(INFO) << "Feature result:";
    std::string outdata = "\n";
    for(auto& data : output) {
        idx++;
        outdata += std::to_string(data) + " ";
        if(idx % 16 == 0) {
            outdata += "\n";
        }
    }
    LOG(INFO) << outdata;

    return 0;
}
