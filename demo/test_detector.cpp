#include <iostream>
#include "detector.h"
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include "logging.h"
#include "timer.h"

using namespace std;
using namespace vision;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << argv[0] << " cfgfile imgfile" << std::endl;
        return -1;
    }
    std::string cfgfile = argv[1];
    std::string imgfile = argv[2];
    cv::Mat image = cv::imread(imgfile);
    cv::Mat img_disp = image.clone();

    Detector* det = new Detector(cfgfile);
    std::map<std::string, std::vector<Box> > boxes_map;
    HighClock clk;
    clk.Start();
    int ret = det->run(image, boxes_map);
    clk.Stop();
    std::cout << "detection run time:" << clk.GetTime() / 1000 << " ms." << std::endl;
    delete det;
    if(ret) {
        std::cout << "Detector run err:" << ret << std::endl;
        return -1;
    }

    for(auto it = boxes_map.begin(); it != boxes_map.end(); ++it) {
        std::cout << it->first << ":" << std::endl;
        std::vector<Box>& boxes = (it->second);
        for(auto& b:boxes) {
            std::cout << "score:" << b.score << " ,x1:" << b.x1 << " ,y1:" << b.y1 << " ,x2:" << b.x2 << " ,y2:" << b.y2 << std::endl;
            cv::Rect rect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
            cv::rectangle(img_disp, rect, cv::Scalar(0,255,0), 2);
        }
    }
    cv::imshow("detector", img_disp);
    cv::waitKey(0);
    return 0;
}
