#ifndef _REIDFEATURE_POST_H_
#define _REIDFEATURE_POST_H_

#include "postprocess.h"

using json=nlohmann::json;

namespace vision {


/**
 * @brief Post-processing for reid feature algorithms
 * 
 */
class ReidFeaturePost: public PostprocOp {
public:
  /**
   * @brief Construct a new ReidFeature Post object
   *
   * @param node
   */
  ReidFeaturePost(const json &node);

  /**
   * @brief setup vision::DLCVOut structure afer forward computataion.
   *
   * By now, \c run just copy \c indata[0] to \c out.feature.dim
   *
   * @param indata result from inference backend
   * @param out target output object
   * @return int 0 when success, 1 otherwise
   */
  virtual int run(const vector<Blob>& indata,
                  const PreprocInfo& info,
                  vision::DLCVOut &out) override;

};


} // namespace vision

#endif // _MULTICLASS_POST_H_
