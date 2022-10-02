#include <iostream>
#include "keypoints_attributes.h"
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include "logging.h"

using namespace std;
using namespace vision;


int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cout << argv[0] << " cfgfile imgfile" << std::endl;
        return -1;
    }

    std::string cfgfile = argv[1];
    std::string imgfile = argv[2];

    auto kps_attr = new vision::KeypointsAttributes(cfgfile);
    std::vector<cv::Point> keypoints;
    std::vector<std::vector<float> > attributes;
    cv::Mat image = cv::imread(imgfile);

    kps_attr->run(image, keypoints, attributes);

    int idx = 0;
    LOG(INFO) << "Keypoints result:";
    std::string outdata = "\n";
    for(auto& data : keypoints) {
        idx++;
        outdata += std::to_string(data.x) + " " + std::to_string(data.y) + "\n";
    }
    LOG(INFO) << outdata;

    LOG(INFO) << "Attributes result:";
    outdata = "\n";
    for(auto& arr : attributes) {
        for(auto& data : arr) {
            outdata += std::to_string(data) + " ";
        }
        outdata += "\n";
    }
    LOG(INFO) << outdata;

    return 0;
}
