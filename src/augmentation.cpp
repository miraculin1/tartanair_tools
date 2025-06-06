#include <chrono> // 头文件
#include <filesystem>
#include <iomanip> // 用于设置输出格式
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <cnpy.h>
#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
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

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr project_pc(const Mat &rgb,
                                                    const Mat &depth) {
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
    return cloud;
  }

  Mat reproj_r_t(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const Mat &depth,
                 const Matrix3f &R, const Vector3f &t, bool inpaint = false) {
    int height = depth.rows;
    int width = depth.cols;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(R);         // 应用旋转矩阵
    transform.translation() = t; // 应用平移向量
    pcl::transformPointCloud(*cloud, *cloud_transformed, transform);

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
                                   const vector<Matrix3f> &delta_rotations,
                                   const vector<Vector3f> &delta_translations,
                                   vector<Mat> &images,
                                   vector<pair<Matrix3f, Vector3f>> &gt_poses,
                                   bool inpaint = false, bool use_pcl = false) {
    auto cloud = project_pc(rgb, depth);
#pragma omp parallel for
    for (size_t i = 0; i < delta_rotations.size(); i++) {
      auto R_tar = delta_rotations[i];
      auto t_tar = delta_translations[i];
      const Matrix3f &dR = delta_rotations[i];
      const Vector3f &dt = delta_translations[i];
      Mat img;
      img = reproj_r_t(cloud, depth, R_tar, t_tar, inpaint);
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
      const cv::Mat &rgb,
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
      std::ostringstream oss0, oss1;
      oss0 << folder_path << "/frame_" << std::setw(3) << std::setfill('0')
           << 2 * i << ".png";
      cv::imwrite(oss0.str(), rgb);

      oss1 << folder_path << "/frame_" << std::setw(3) << std::setfill('0')
           << 2 * i + 1 << ".png";
      cv::imwrite(oss1.str(), imgs[i]);

      // 转换旋转矩阵到四元数
      const Eigen::Matrix3f &R = gt_poses[i].first;
      const Eigen::Vector3f &t = gt_poses[i].second;
      Eigen::Quaternionf q(R);

      // 按 tx ty tz qx qy qz qw 格式写一行
      pose_file << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0
                << " " << 1 << std::endl;
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

  // 生成轨迹增量
  vector<Matrix3f> delta_Rs;
  vector<Vector3f> delta_ts;
  augmenter.generate_delta_rt_6directions(6, 0.1f, 0.01f, 0.01f, delta_Rs,
                                          delta_ts);
  auto t_gen_delta_fin = std::chrono::high_resolution_clock::now();

  vector<Mat> imgs;
  vector<pair<Matrix3f, Vector3f>> gt_poses;

  // 这里可切换 use_pcl = true / false
  bool use_pcl = false;

  augmenter.generate_trajectory_with_gt(rgb, depth, delta_Rs, delta_ts, imgs,
                                        gt_poses, true, use_pcl);
  auto t_aug_fin = std::chrono::high_resolution_clock::now();
  auto t_stop = std::chrono::high_resolution_clock::now();

  augmenter.saveImgsAndPoses("./P000/image_left", imgs, rgb, gt_poses);
  // for (size_t i = 0; i < imgs.size(); i++) {
  // imshow("Augmented Image", imgs[i]);
  // int key = waitKey(0);
  // if (key == 27)
  // break; // ESC退出
  // }

  printDuration(t_start, t_stop);
  printDuration(t_gen_delta_fin, t_aug_fin);
  return 0;
}
