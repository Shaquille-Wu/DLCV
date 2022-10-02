#include <iostream>
#include "multiclass.h"
#include "opencv2/opencv.hpp"
#include <getopt.h>
#include <sys/time.h>
#include "logging.h"

using namespace std;
using namespace vision;

using vision::Multiclass;

static std::string trim_string(const std::string& src)
{
    int    str_len     = src.length();
    int    start_pos   = 0;
    int    end_pos     = 0;
    int    i           = 0;
    char*  trim_str    = new char[str_len + 1];
    memset(trim_str, 0, str_len + 1);
    for(i = 0 ; i < str_len ; i ++)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            start_pos = i;
            break;
        }
    }
    for(i = str_len - 1 ; i >= 0 ; i --)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            end_pos = i;
            break;
        }
    }

    int trim_len = end_pos - start_pos + 1;
    for(i = 0 ; i < trim_len ; i ++)
        trim_str[i] = (src.c_str())[i + start_pos];

    std::string  result = std::string(trim_str);
    delete[] trim_str;

    return result;
}

int main(int argc, char* argv[]){
    std::string   cfgfile;
    std::string   imgfile;
    std::string   run_loop  = "";
    int           opt       = 0;
    while ((opt = getopt(argc, argv, "hj:i:n:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
               LOG(INFO) 
                        << "\nDESCRIPTION:\n"
                        << "  -j  confige file(.json file).\n"
                        << "  -i  image file.\n"
                        << "  -n  run loop count.\n"                      
                        << "\n";
                std::exit(0);
                break;
            case 'j':
                cfgfile        = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;                
            case 'i':
                imgfile        = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;            
            case 'n':
                run_loop       = atoi(trim_string((nullptr == optarg ? std::string("") : std::string(optarg))).c_str());
                break;                        
            default:
                LOG(INFO) << "Invalid parameter specified.";
                std::exit(-1);
        }
    }    

    int loop_count = atoi(run_loop.c_str());
    if(loop_count <= 0)
        loop_count = 1;

    std::cout << "cfgfile: " << cfgfile << std::endl;
    std::cout << "imgfile: " << imgfile << std::endl;
    std::cout << "runloop: " << loop_count << std::endl;


    auto output = Multiclass::output_type();
    auto multiclass = new Multiclass(cfgfile);
    cv::Mat image = cv::imread(imgfile);

    int               i               = 0;
    struct  timeval   tv_start        = { 0, 0 } ;
    struct  timeval   tv_end          = { 0, 0 } ;
    long long int     inference_time  = 0;

    gettimeofday(&tv_start,0);  
    for(i = 0 ; i < loop_count ; i ++)  
    {
        output.clear();   
        multiclass->run(image, output);
    }
    gettimeofday(&tv_end,0);
    inference_time = 1000000 * (tv_end.tv_sec-tv_start.tv_sec)+ tv_end.tv_usec-tv_start.tv_usec;
    LOG(INFO) << "detect time eplased: " << inference_time;    

    LOG(INFO) << "Multiclass result:";
    std::string outdata;
    for(auto& arr : output) {
        for(auto& data : arr) {
            outdata += std::to_string(data) + " ";
        }
        outdata += "\n";
    }
    LOG(INFO) << outdata;

    return 0;
}
