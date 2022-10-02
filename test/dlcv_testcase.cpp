#include <gtest/gtest.h>
#include "dlcv_testcase.h"
#include "logging.h"


DlcvTestCase::DlcvTestCase(const char* json_result){

    auto  case_cfg = load_json(json_result);
    EXPECT_GT(case_cfg.size(), 0);

    for (json::iterator it = case_cfg.begin(); it != case_cfg.end(); ++it) {
        std::string image_path = (*it)["input"];

        vision::DLCVOut out;
        if (it->contains("boxes")) {
            out.has_boxes = true;
            auto box_result = (*it)["boxes"];
            for (json::iterator box_it = box_result.begin(); box_it != box_result.end(); ++box_it) {
                vision::Box new_box;
                new_box.score = (*box_it)["score"];
                new_box.x1 = (*box_it)["x1"];
                new_box.x2 = (*box_it)["x2"];
                new_box.y1 = (*box_it)["y1"];
                new_box.y2 = (*box_it)["y2"];

                std::string type = (*box_it)["type"];
                insert_map(out.boxes_map, type, new_box);
            }
        }

        if (it->contains("feature")) {
            out.has_feature = true;
            auto feature = (*it)["feature"];
            for (json::iterator feature_it = feature.begin(); feature_it != feature.end(); ++feature_it) {
                out.feature.push_back((float)*feature_it);
            }
        }

        if (it->contains("multiclass")) {
            out.has_multiclass = true;
            auto multiclass = (*it)["multiclass"];
            std::vector<float> class_list;
            for (json::iterator class_it = multiclass.begin(); class_it != multiclass.end(); ++class_it) {
                class_list.push_back((float)*class_it);
            }
            out.multiclass.push_back(class_list);

        }

        if (it->contains("keypoints")) {
            out.has_keypoints = true;
            auto keypoints = (*it)["keypoints"];
            for (json::iterator point = keypoints.begin(); point != keypoints.end(); ++point) {
                cv::Point new_point((*point)["x"], (*point)["y"]);
                out.keypoints.push_back(new_point);
            }
        }

        if (it->contains("featuremaps")) {
            out.has_featuremaps = true;

            vision::FeatureMap new_item;
            auto featuremaps = (*it)["featuremaps"];
            for (json::iterator featuremap = featuremaps.begin(); featuremap != featuremaps.end(); ++featuremap) {
                auto shape = (*featuremap)["shape"];
                auto data = (*featuremap)["data"];
                auto value_type = (*featuremap)["value_type"];

                // shape
                int element_count = 1;
                for (auto shape_item : shape) {
                    new_item.shape.push_back(shape_item);
                    element_count *= (int)shape_item;
                }

                // value_type
                if (value_type == "int") {
                    new_item.type = vision::ElementType::I32;
                }
                else {
                    new_item.type = vision::ElementType::F32;
                }

                // data
                std::shared_ptr<unsigned char> managed;
                managed.reset(new unsigned char[element_count * sizeof(float)]);
                EXPECT_EQ(element_count, data.size());
                for (int i = 0; i < data.size(); i++) {
                        ((float*)managed.get())[i] = (float)data[i];
                }
                new_item.data = managed;

                out.featuremaps.push_back(new_item);
                feature_map_holder.push_back(managed);
            }
        }
        case_list.emplace_back(image_path, out);
    }
}

void DlcvTestCase::CheckOutput(vision::DLCVOut &currentOut, vision::DLCVOut &historyOut) {

    EXPECT_EQ(currentOut.has_boxes, historyOut.has_boxes );
    EXPECT_EQ(currentOut.has_feature, historyOut.has_feature);
    EXPECT_EQ(currentOut.has_multiclass, historyOut.has_multiclass);
    EXPECT_EQ(currentOut.has_keypoints, historyOut.has_keypoints);
    EXPECT_EQ(currentOut.has_featuremaps, historyOut.has_featuremaps);

    if (currentOut.has_boxes) {
        EXPECT_EQ(currentOut.boxes_map.size(), historyOut.boxes_map.size());
        for (auto currentItem = currentOut.boxes_map.begin(); currentItem != currentOut.boxes_map.end(); currentItem++) {

            std::string labelName = currentItem->first;
            auto history_item = historyOut.boxes_map.find(labelName);
            EXPECT_TRUE(history_item != historyOut.boxes_map.end());

            std::vector<vision::Box> &currentBoxList = currentItem->second;
            std::vector<vision::Box> &historyBoxList = history_item->second;
            EXPECT_EQ(currentBoxList.size(), historyBoxList.size());

            for (int i = 0; i < currentBoxList.size(); i++) {
                const vision::Box &box1 = currentBoxList[i];
                const vision::Box &box2 = historyBoxList[i];

                EXPECT_NEAR(box1.x1, box2.x1, 0.1);
                EXPECT_NEAR(box1.y1, box2.y1, 0.1);
                EXPECT_NEAR(box1.x2, box2.x2, 0.1);
                EXPECT_NEAR(box1.y2, box2.y2, 0.1);
                EXPECT_NEAR(box1.score, box2.score, 0.1);
            }
        }
    }

    if (currentOut.has_feature) {
        EXPECT_EQ(currentOut.feature.size(), historyOut.feature.size());

        printf("feature\n");
        for (int i = 0; i < currentOut.feature.size(); i++) {
                printf("%f,\n", currentOut.feature[i]);
        }
        printf("-------\n");

        for (int i = 0; i < currentOut.feature.size(); i++) {
            EXPECT_NEAR(currentOut.feature[i], historyOut.feature[i], 0.01);
        }
    }

    if (currentOut.has_multiclass) {
        EXPECT_EQ(currentOut.multiclass.size(), historyOut.multiclass.size());

        printf("multiclass\n");
        for (int i = 0; i < currentOut.multiclass.size(); i++) {
            std::vector<float> &current_class_list = currentOut.multiclass[i];
            std::vector<float> &history_class_list = historyOut.multiclass[i];
            for (int j = 0; j < current_class_list.size(); j++) {
                printf("%f,\n", current_class_list[j]);
            }
        }
        printf("-------\n");

        for (int i = 0; i < currentOut.multiclass.size(); i++) {
            std::vector<float> &current_class_list = currentOut.multiclass[i];
            std::vector<float> &history_class_list = historyOut.multiclass[i];
            for (int j = 0; j < current_class_list.size(); j++) {
                EXPECT_NEAR(current_class_list[j], history_class_list[j], 0.01);
            }
        }
    }

    if (currentOut.has_keypoints) {
        EXPECT_EQ(currentOut.keypoints.size(), historyOut.keypoints.size());

        printf("keypoints\n");
        for (int i = 0; i < currentOut.keypoints.size(); i++) {
            printf("{\"x\":%d,\"y\":%d},\n", currentOut.keypoints[i].x, currentOut.keypoints[i].y);
        }
        printf("-------\n");

        for (int i = 0; i < currentOut.feature.size(); i++) {
            EXPECT_NEAR(currentOut.feature[i], historyOut.feature[i], 0.01);
        }
    }

    if (currentOut.has_featuremaps) {
        EXPECT_EQ(currentOut.featuremaps.size(), historyOut.featuremaps.size());

        for (int i = 0; i < currentOut.featuremaps.size(); i++) {

            int elementCount = 1;
            printf("shape:%d\n",i);
            for (int j = 0; j < currentOut.featuremaps[i].shape.size(); j++) {
                printf("%d,\n", currentOut.featuremaps[i].shape[j]);
                elementCount *= currentOut.featuremaps[i].shape[j];
                EXPECT_EQ(currentOut.featuremaps[i].shape[j], historyOut.featuremaps[i].shape[j]);
            }

            printf("data:%d\n",i);
            auto current_ptr = currentOut.featuremaps[i].data.lock();
            auto history_ptr = historyOut.featuremaps[i].data.lock();
            for (int j = 0; j < elementCount; j++) {
                if (historyOut.featuremaps[i].type == vision::ElementType::I32) {
                    printf("%d,\n", ((int*)current_ptr.get())[j]);
                }
                else {
                    printf("%f,\n", ((float*)current_ptr.get())[j]);
                }
            }
            printf("-------\n");

            for (int j = 0; j < currentOut.featuremaps[i].shape.size(); j++) {
                EXPECT_EQ(currentOut.featuremaps[i].shape[j], historyOut.featuremaps[i].shape[j]);
            }
            for (int j = 0; j < elementCount; j++) {
                if (historyOut.featuremaps[i].type == vision::ElementType::I32) {
                    float current_value = ((int*)current_ptr.get())[j];
                    float history_value = ((float*)history_ptr.get())[j];
                    EXPECT_NEAR(current_value, history_value, 0.01);

                    //if (current_value != history_value) {
                    //    printf("diff:%d %f,%f\n", j, current_value, history_value);
                    //}
                }
                else {
                    EXPECT_NEAR(((float*)current_ptr.get())[j], ((float*)history_ptr.get())[j], 0.01);
                }
            }
        }
    }
}


json DlcvTestCase::load_json(const char* json_result) {
    json cfg;
    try {
        std::ifstream inJsonFile(json_result);
        if (!inJsonFile) {
            LOG(ERROR) << " Error: json file open failed: " << json_result;
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

vector<vision::Box>& DlcvTestCase::insert_map(std::map<string, std::vector<vision::Box> >& d, string key, vision::Box& new_val) {
    std::map<string, std::vector<vision::Box> >::iterator it = d.find(key);
    if (it != d.end()) {
        it->second.push_back(new_val);
    }
    else {
        std::vector<vision::Box> v;
        v.push_back(new_val);
        d.insert(make_pair(key, v));
        return d[key];
    }
    return it->second;
}
