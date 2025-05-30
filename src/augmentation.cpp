#include <chrono> // 头文件
#include <filesystem>
#include <iomanip> // 用于设置输出格式
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <cnpy.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace cv;
using namespace Eigen;
using namespace std;
namespace fs = std::filesystem;

class TarTanAirAugmentation {
public:
  float fx = 320.f;
  float fy = 320.f;
  float cx = 320.f;
  float cy = 240.f;

  TarTanAirAugmentation() {}

  // 传统逐像素投影
  Mat reproj_r_t(const Mat &rgb, const Mat &depth, const Matrix3f &R,
                 const Vector3f &t, bool inpaint = false) {
    int height = depth.rows;
    int width = depth.cols;
    Mat aug_img = Mat::zeros(rgb.size(), rgb.type());
    Mat depth_buffer = Mat(height, width, CV_32FC1, Scalar(1e6f));
    for (int v = 0; v < height; v++) {
      for (int u = 0; u < width; u++) {
        float Z = depth.at<float>(v, u);
        if (Z <= 0)
          continue;
        float X = (u - cx) * Z / fx;
        float Y = (v - cy) * Z / fy;
        Vector3f pt3d(X, Y, Z);
        Vector3f pt3d_trans = R.transpose() * pt3d + t;
        float x_proj = (pt3d_trans(0) * fx) / pt3d_trans(2) + cx;
        float y_proj = (pt3d_trans(1) * fy) / pt3d_trans(2) + cy;
        if (x_proj < 0 || x_proj >= width || y_proj < 0 || y_proj >= height)
          continue;
        int x_round = static_cast<int>(round(x_proj));
        int y_round = static_cast<int>(round(y_proj));
        if (x_round < 0 || x_round >= width || y_round < 0 || y_round >= height)
          continue;
        float z = pt3d_trans(2);
        float &buf_z = depth_buffer.at<float>(y_round, x_round);
        if (z < buf_z) {
          aug_img.at<Vec3b>(y_round, x_round) = rgb.at<Vec3b>(v, u);
          buf_z = z;
        }
      }
    }
    if (inpaint) {
      Mat gray_mask;
      cv::inRange(aug_img, Scalar(0, 0, 0), Scalar(0, 0, 0), gray_mask);
      Mat inpainted;
      cv::inpaint(aug_img, gray_mask, inpainted, 3, INPAINT_TELEA);
      return inpainted;
    }
    return aug_img;
  }

  // 使用PCL实现的投影函数
  Mat reproj_r_t_pcl(const Mat &rgb, const Mat &depth, const Matrix3f &R,
                     const Vector3f &t, bool inpaint = false) {
    int height = depth.rows;
    int width = depth.cols;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(width * height);

    for (int v = 0; v < height; v++) {
      const float *depth_ptr = depth.ptr<float>(v);
      const Vec3b *rgb_ptr = rgb.ptr<Vec3b>(v);
      for (int u = 0; u < width; u++) {
        int idx = v * width + u;
        float Z = depth_ptr[u];
        pcl::PointXYZRGB &pt = cloud->points[idx];
        if (Z <= 0) {
          pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
          pt.r = pt.g = pt.b = 0;
          continue;
        }
        float X = (u - cx) * Z / fx;
        float Y = (v - cy) * Z / fy;
        pt.x = X;
        pt.y = Y;
        pt.z = Z;
        pt.b = rgb_ptr[u][0];
        pt.g = rgb_ptr[u][1];
        pt.r = rgb_ptr[u][2];
      }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_transformed->width = cloud->width;
    cloud_transformed->height = cloud->height;
    cloud_transformed->is_dense = false;
    cloud_transformed->points.resize(cloud->points.size());

    for (size_t i = 0; i < cloud->points.size(); i++) {
      const auto &pt_in = cloud->points[i];
      auto &pt_out = cloud_transformed->points[i];
      if (!pcl::isFinite(pt_in)) {
        pt_out = pt_in;
        continue;
      }
      Vector3f p(pt_in.x, pt_in.y, pt_in.z);
      Vector3f p_trans = R.transpose() * p + t;
      pt_out.x = p_trans.x();
      pt_out.y = p_trans.y();
      pt_out.z = p_trans.z();
      pt_out.r = pt_in.r;
      pt_out.g = pt_in.g;
      pt_out.b = pt_in.b;
    }

    Mat aug_img = Mat::zeros(height, width, CV_8UC3);
    Mat depth_buffer(height, width, CV_32FC1, Scalar(1e6f));

    for (size_t i = 0; i < cloud_transformed->points.size(); i++) {
      const auto &pt = cloud_transformed->points[i];
      if (!pcl::isFinite(pt))
        continue;
      if (pt.z <= 0)
        continue;
      float x_proj = (pt.x * fx) / pt.z + cx;
      float y_proj = (pt.y * fy) / pt.z + cy;
      int x = static_cast<int>(round(x_proj));
      int y = static_cast<int>(round(y_proj));
      if (x < 0 || x >= width || y < 0 || y >= height)
        continue;
      float &buf_z = depth_buffer.at<float>(y, x);
      if (pt.z < buf_z) {
        buf_z = pt.z;
        aug_img.at<Vec3b>(y, x)[0] = pt.b;
        aug_img.at<Vec3b>(y, x)[1] = pt.g;
        aug_img.at<Vec3b>(y, x)[2] = pt.r;
      }
    }

    if (inpaint) {
      Mat gray_mask;
      cv::inRange(aug_img, Scalar(0, 0, 0), Scalar(0, 0, 0), gray_mask);
      Mat inpainted;
      cv::inpaint(aug_img, gray_mask, inpainted, 3, INPAINT_TELEA);
      return inpainted;
    }
    return aug_img;
  }

  void generate_trajectory_with_gt(const Mat &rgb, const Mat &depth,
                                   const Matrix3f &R0, const Vector3f &t0,
                                   const vector<Matrix3f> &delta_rotations,
                                   const vector<Vector3f> &delta_translations,
                                   vector<Mat> &images,
                                   vector<pair<Matrix3f, Vector3f>> &gt_poses,
                                   bool inpaint = false, bool use_pcl = false) {
    Matrix3f R_tar = R0;
    Vector3f t_tar = t0;
    for (size_t i = 0; i < delta_rotations.size(); i++) {
      const Matrix3f &dR = delta_rotations[i];
      const Vector3f &dt = delta_translations[i];
      R_tar = dR * R0;
      t_tar = dR * t0 + dt;
      Mat img;
      if (use_pcl)
        img = reproj_r_t_pcl(rgb, depth, R_tar, t_tar, inpaint);
      else
        img = reproj_r_t(rgb, depth, R_tar, t_tar, inpaint);
      images.push_back(img);
      gt_poses.emplace_back(R_tar, t_tar);
    }
  }

  // 你之前的增量生成函数，保持不变
  void
  generate_delta_rt_6directions(int n_steps, float step_length,
                                float rot_noise_std, float trans_noise_std,
                                std::vector<Matrix3f> &delta_rotations,
                                std::vector<Vector3f> &delta_translations) {
    std::vector<Vector3f> directions = {Vector3f(1, 0, 0), Vector3f(-1, 0, 0),
                                        Vector3f(0, 1, 0), Vector3f(0, -1, 0),
                                        Vector3f(0, 0, 1), Vector3f(0, 0, -1)};

    std::default_random_engine generator;
    std::normal_distribution<float> noise_trans(0.0f, trans_noise_std);
    std::normal_distribution<float> noise_rot(0.0f, rot_noise_std);
    std::normal_distribution<float> noise_axis(0.0f, 1.0f);

    for (int i = 0; i < n_steps; i++) {
      int dir_idx = i % 6;
      Vector3f base_dir = directions[dir_idx];
      Vector3f trans_noise_vec(noise_trans(generator), noise_trans(generator),
                               noise_trans(generator));
      Vector3f delta_t = base_dir * step_length + trans_noise_vec;
      Vector3f axis(noise_axis(generator), noise_axis(generator),
                    noise_axis(generator));
      if (axis.norm() < 1e-6)
        axis = Vector3f(1, 0, 0);
      axis.normalize();
      float angle = noise_rot(generator);
      AngleAxisf aa(angle, axis);
      Matrix3f delta_R = aa.toRotationMatrix();
      delta_rotations.push_back(delta_R);
      delta_translations.push_back(delta_t);
    }
  }

  void saveImgsAndPoses(
      const std::string &folder_path, const std::vector<cv::Mat> &imgs,
      const std::vector<std::pair<Eigen::Matrix3f, Eigen::Vector3f>>
          &gt_poses) {
    namespace fs = std::filesystem;
    fs::create_directories(folder_path);

    std::ofstream pose_file(folder_path + "/poses.txt");
    if (!pose_file.is_open()) {
      std::cerr << "Failed to open pose file for writing!" << std::endl;
      return;
    }

    pose_file << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < imgs.size(); ++i) {
      // 保存图像
      std::ostringstream oss;
      oss << folder_path << "/frame_" << std::setw(3) << std::setfill('0') << i
          << ".png";
      cv::imwrite(oss.str(), imgs[i]);

      // 转换旋转矩阵到四元数
      const Eigen::Matrix3f &R = gt_poses[i].first;
      const Eigen::Vector3f &t = gt_poses[i].second;
      Eigen::Quaternionf q(R);

      // 按 tx ty tz qx qy qz qw 格式写一行
      pose_file << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " "
                << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    pose_file.close();
    std::cout << "Saved " << imgs.size() << " images and poses to "
              << folder_path << std::endl;
  }
};

cv::Mat readDepthNpy(const std::string &filename) {
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  if (arr.shape.size() != 2) {
    throw std::runtime_error("Depth npy must be 2D");
  }
  int height = arr.shape[0];
  int width = arr.shape[1];
  float *data_ptr = arr.data<float>();
  cv::Mat depth(height, width, CV_32F);
  memcpy(depth.data, data_ptr, height * width * sizeof(float));
  return depth;
}

using namespace std::chrono;
void printDuration(const high_resolution_clock::time_point &start,
                   const high_resolution_clock::time_point &end) {
  auto duration = duration_cast<microseconds>(end - start);
  double ms = duration.count() / 1000.0;
  std::cout << "Elapsed time: " << std::fixed << std::setprecision(3) << ms
            << " ms" << std::endl;
}

int main(int argc, char **argv) {
  auto t_start = std::chrono::high_resolution_clock::now();
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <rgb_image_path> <depth_npy_path>"
              << std::endl;
    return -1;
  }
  std::string rgb_path = argv[1];
  std::string depth_path = argv[2];
  Mat rgb = imread(rgb_path);
  if (rgb.empty()) {
    std::cerr << "Failed to load RGB image: " << rgb_path << std::endl;
    return -1;
  }
  Mat depth;
  try {
    depth = readDepthNpy(depth_path);
  } catch (const std::exception &e) {
    std::cerr << "Failed to load depth npy: " << e.what() << std::endl;
    return -1;
  }
  if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
    std::cerr << "RGB and depth size mismatch" << std::endl;
    return -1;
  }
  auto t_r_fin = std::chrono::high_resolution_clock::now();

  TarTanAirAugmentation augmenter;

  // 初始姿态
  float theta = 2.0f * M_PI / 180.0f;
  Matrix3f R0 = AngleAxisf(theta, Vector3f(0, 1, 0)).toRotationMatrix();
  Vector3f t0(0.0f, 0.0f, 0.0f);

  // 生成轨迹增量
  vector<Matrix3f> delta_Rs;
  vector<Vector3f> delta_ts;
  augmenter.generate_delta_rt_6directions(16, 0.1f, 0.01f, 0.01f, delta_Rs,
                                          delta_ts);
  auto t_gen_delta_fin = std::chrono::high_resolution_clock::now();

  vector<Mat> imgs;
  vector<pair<Matrix3f, Vector3f>> gt_poses;

  // 这里可切换 use_pcl = true / false
  bool use_pcl = false;

  augmenter.generate_trajectory_with_gt(rgb, depth, R0, t0, delta_Rs, delta_ts,
                                        imgs, gt_poses, true, use_pcl);
  auto t_aug_fin = std::chrono::high_resolution_clock::now();
  auto t_stop = std::chrono::high_resolution_clock::now();

  augmenter.saveImgsAndPoses("./res", imgs, gt_poses);
  for (size_t i = 0; i < imgs.size(); i++) {
    imshow("Augmented Image", imgs[i]);
    int key = waitKey(0);
    if (key == 27)
      break; // ESC退出
  }

  printDuration(t_start, t_stop);
  printDuration(t_gen_delta_fin, t_aug_fin);
  return 0;
}
