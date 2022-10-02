#pragma once
#include "dlcv.h"               // for ElementType, F32
#include "json.hpp"               // for ElementType, F32
using json = nlohmann::json;

class DlcvTestCase {
public:
    DlcvTestCase(const char* json_result);
    void CheckOutput(vision::DLCVOut &currentOut, vision::DLCVOut &historyOut);

    std::list< std::pair<std::string, vision::DLCVOut> > case_list;

private:

    vector<vision::Box>& insert_map(std::map<string, std::vector<vision::Box> >& d, string key, vision::Box& new_val);
    json load_json(const char* json_result);

    std::list< std::shared_ptr<unsigned char> > feature_map_holder;
};

