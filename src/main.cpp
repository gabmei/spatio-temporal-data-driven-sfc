#include<iostream>
#include<vector>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "data_driven.hpp"
#include "prim.hpp"
#include "opencv_util.hpp"

const std::string FILENAME = "images/image_test2.png";

int main() {
  cv::Mat read_image = opencv_util::open_8bit_image(FILENAME);
  cv::imshow(FILENAME, read_image);
  auto conv_img = opencv_util::convert_8bit_image_to_vector(read_image);

  DataDrivenDistance<double, int> distance_calc(conv_img, 0.3, 4);
  Prim<double, int> prim(read_image.rows, read_image.cols);
  std::vector<std::pair<int, int>> ord = prim.run(distance_calc);

  cv::Mat heatmap_img = opencv_util::heatmap_image(read_image.rows, read_image.cols, ord);
  cv::imshow(FILENAME + " heatmap", heatmap_img);
  
  cv::Mat path_img = opencv_util::path_image(read_image, ord);
  cv::imshow(FILENAME + " path", path_img);
  cv::waitKey(0);
  return 0;
}