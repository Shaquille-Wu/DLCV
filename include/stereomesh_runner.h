#ifndef _STEREO_MESH_H_
#define _STEREO_MESH_H_

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "dlcv.h"


namespace vision{

class StereoMeshRunner {
public:
  //! StereoMeshRunner use std::vector<float> as output type
  using output_type = std::vector<FeatureMap>;

  /**
   * @brief Construct a new StereoMesh object via JSON config
   *
   * @param config_file path to config file.
   */
  explicit StereoMeshRunner(std::string config_file);

  ~StereoMeshRunner();

  /**
   * @brief Run StereoMesh model
   * 
   * @param images Input images.
   * @param featuremaps reference to output
   * @return int 
   */
  int run(std::vector<cv::Mat*> &images, std::vector<FeatureMap>& featuremaps);

private:
  class Impl;
  Impl* pimpl;
};

}

#endif