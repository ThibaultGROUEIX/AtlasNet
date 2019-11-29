#ifndef _VIRTUAL_SCANNER_
#define _VIRTUAL_SCANNER_

#include <vector>
#include <string>
#include <Eigen/Dense>

#include "points.h"

using std::string;
using std::vector;
using Eigen::Vector3f;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

class VirtualScanner {
 public:
  bool scanning(const string& filename, int view_num, bool flags, bool normalize);
  bool save_binary(const string& filename);
  bool save_ply(const string& filename);

 protected:
  bool save_binary_legacy(const string& filename);
  void calc_views();

 protected:
  MatrixXf V_;
  MatrixXi F_;
  MatrixXf N_;
  Vector3f bbMin_, bbMax_;
  vector<Vector3f> view_center_;
  vector<Vector3f> view_dir_;
  vector<Vector3f> dx_;
  vector<Vector3f> dy_;

  vector<float> pts_;
  vector<float> normals_;
  vector<int> flags_; // indicate whether the normal is reversed

  Points point_cloud_;

  const int resolution_ = 127;
  const int total_view_num_ = 14;

};

#endif // _VIRTUAL_SCANNER_
