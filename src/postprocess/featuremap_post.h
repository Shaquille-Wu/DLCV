#ifndef _FEATUREMAP_POST_H_
#define _FEATUREMAP_POST_H_

#include <array>          // for array
#include <memory>         // for unique_ptr
#include <vector>         // for vector

#include "json.hpp"       // for json
#include "postprocess.h"  // for PostprocOp

namespace vision { class Blob; }
namespace vision { class DLCVOut; }
namespace vision { struct PreprocInfo; }

using json = nlohmann::json;

namespace vision {

template<typename T>
struct option {
  bool has_value_;
  T value_;

  constexpr explicit option() : has_value_{false} {}
  explicit option(const T &value) : has_value_{true}, value_{value} {}

  constexpr bool is_none() const { return not has_value_; }
  constexpr bool is_some() const { return has_value_; }
  static constexpr option none() { return option<T>(); }
  const T &as_ref() const { return value_; }
};

enum FeaturemapPostMode {
  kUNKNOWN = 0,
  kCOPY = 1, // DEPRECATED since 2020/06/22
  kRECOLOR = 2,
  kSHARED = 3
};

/**
 * @brief Post-processing for featuremap extractor algorithms, such as segmentation, eco featuremap
 * 
 */
class FeaturemapPost: public PostprocOp {
public:
  /**
   * @brief Construct a new Featuremap Post object
   *
   * @param node
   */
  FeaturemapPost(const json &node);

  /**
   * @brief setup vision::DLCVOut structure afer forward computataion.
   *
   * This function:
   * 1. call \c select to decide with input \c Blob to use
   * 2. call \c execute to process selected \c Blob
   * 3. \c emplace_back above Featuremap to \c out
   * 
   * @param indata result from inference backend
   * @param names name of each \c Blob in \c indata
   * @param out target output object
   * @return 0 when success, non-zero otherwise
   */
  virtual int run(const vector<Blob> &indata,
                  const vector<string> &names,
                  const PreprocInfo& info,
                  vision::DLCVOut &out) override;

  /**
   * @brief setup vision::DLCVOut structure afer forward computataion.
   * @deprecated since 2021/01/04
   *
   * This function:
   * 1. call \c execute to process first \c Blob in \c indata
   * 2. \c emplace_back above Featuremap to \c out
   *
   * @param indata result from inference backend
   * @param out target output object
   * @return int 0 when success, 1 otherwise
   */
  virtual int run(const vector<Blob> &indata,
                  const PreprocInfo& info,
                  vision::DLCVOut &out) override;

  /**
   * @brief setup vision::DLCVOut structure afer forward computataion.
   */
  int execute(const Blob &blob, vision::DLCVOut &out);

  /**
   * @brief select \c Blob to process by following rules:
   * 
   * 1. if names.empty() or source.is_none(), returns 0.
   * 2. if source.as_ref() == names[i], returns i.
   * 3. retures names.size()
   * 
   * @param names names of blobs
   * @return [0, names.size()) if found, names.size() otherwise
   */
  size_t select(const vector<string> &names) const;

  //! Mode. Configured by "mode" in JSON.
  FeaturemapPostMode mode = kUNKNOWN;

  //! Whether to apply HWC->CHW transform on final \c FeatureMap.
  //! Configured by "hwc2chw" in JSON.
  bool hwc2chw_ = false;

  //! Colormap to use.
  //! Only significative when mode == kRECOLOR.
  //! Configured by "colormap" in JSON.
  std::unique_ptr<std::vector<std::array<float, 3>>> colormap = nullptr;

  //! Buffer to use. Only significative when (hwc2chw or mode == kRECOLOR).
  Blob buffer_;

  //! Optional. Name of source blob. Configured by "input" in JSON.
  //! clang >= 5 compiles C++17. replace it with std::optional some day
  option<string> source;
};

} // namespace vision

#endif // _FEATUREMAP_POST_H is not defined 
