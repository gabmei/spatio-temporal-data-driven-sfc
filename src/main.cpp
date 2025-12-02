#include<iostream>
#include <filesystem>
#include<vector>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "data_driven.hpp"
#include "prim.hpp"
#include "opencv_util.hpp"

void process_images(const std::vector<std::filesystem::path>& image_paths, const std::filesystem::path output_path, bool is_grayscale = false, int pixel_width = 200) {
  cv::Mat output_img;
  int rows = -1, cols = -1;
  for(int i = 0, len = (int)image_paths.size(); i < len; ++i) {
    cv::Mat read_img = opencv_util::open_8bit_image(image_paths[i], is_grayscale);

    if(i == 0) {
      rows = read_img.rows * read_img.cols;
      cols = len * pixel_width;

      output_img = cv::Mat::zeros(rows, cols, is_grayscale ? CV_8UC1 : CV_8UC3);
    }
    auto conv_img = opencv_util::convert_8bit_image_to_vector(read_img);
    DataDrivenDistance<double, int> distance_calc(conv_img, 0.03, 10);
    Prim<double, int> prim(read_img.rows, read_img.cols);
    std::vector<std::pair<int, int>> ord = prim.run(distance_calc);

    opencv_util::process_image(output_img, read_img, ord, i, pixel_width);

    cv::Mat heatmap_img = opencv_util::heatmap_image(read_img.rows, read_img.cols, ord);
    cv::Mat path_img = opencv_util::path_image(read_img, ord);
    
    cv::imwrite(output_path / ("heatmap_" + image_paths[i].filename().string()), heatmap_img);
    cv::imwrite(output_path / ("img_" + image_paths[i].filename().string()), path_img);
  }
  cv::imwrite(output_path / "timeline.png", output_img);
}


int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << "<path_folder>" << std::endl;
    return 1; // Return an error code
  }
  // reads images in the path folder
  std::filesystem::path input_path = argv[1];
  std::filesystem::path output_path = "out";
  std::filesystem::create_directories(output_path);
  std::vector<std::filesystem::path> all_paths;
  for(const auto& img_path : std::filesystem::directory_iterator(input_path)) {
    all_paths.emplace_back(img_path);
    if(img_path.is_regular_file()) {
      all_paths.emplace_back(img_path.path());
    }
  }
  std::sort(all_paths.begin(), all_paths.end());

  process_images(all_paths, output_path, false);
  
  // if (argc < 2) {
  //   std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
  //   return 1; // Return an error code
  // }
  // // First argument as the filename
  // std::string filename = argv[1];
  // cv::Mat read_image = opencv_util::open_8bit_image(filename, true);
  // cv::imshow(filename, read_image);
  // auto conv_img = opencv_util::convert_8bit_image_to_vector(read_image);

  // DataDrivenDistance<double, int> distance_calc(conv_img, 0.3, 4);
  // Prim<double, int> prim(read_image.rows, read_image.cols);
  // std::vector<std::pair<int, int>> ord = prim.run(distance_calc);

  // cv::Mat heatmap_img = opencv_util::heatmap_image(read_image.rows, read_image.cols, ord);
  // cv::imshow("heatmap_" + filename, heatmap_img);
  // cv::imwrite("heatmap.png", heatmap_img);
  // cv::Mat path_img = opencv_util::path_image(read_image, ord);
  // cv::imshow("path_" + filename, path_img);
  // cv::imwrite("path.png", path_img);
  // cv::waitKey(0);
  return 0;
}