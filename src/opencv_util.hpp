#ifndef OPENCV_UTIL_HPP
#define OPENCV_UTIL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <utility>
#include <iostream>

/**
 * @namespace opencv_util
 * @brief A utility namespace for common OpenCV operations used for this Space-Filling Curve application.
 */
namespace opencv_util {
  const cv::Vec3f BLUE_COLOR(255.0f, 0.0f, 0.0f);
  const cv::Vec3f YELLOW_COLOR(0.0f, 255.0f, 255.0f);

  /**
   * @brief Loads an 8bit image, either grayscale or rgb 
   */
  cv::Mat open_8bit_image(const std::string& filepath, bool is_grayscale = false) {
    cv::Mat img;
    if(is_grayscale) {
      img = cv::imread(filepath, cv::ImreadModes::IMREAD_GRAYSCALE);
    } else {
      img = cv::imread(filepath, cv::ImreadModes::IMREAD_COLOR);
    }

    if (img.empty()) {
      std::cerr << "Error: Could not open or find the image at: " << filepath << std::endl;
    }
    return img;
  }

  /**
   * @brief Draws a heatmap on an image based on the Space-Filling Curve.
   *
   * @param rows The total number of rows
   * @param cols The total number of columns
   * @param ord A vector of (row, col) pairs defining the order of pixels to color.
   * It should be the whole pixels of the image.
   * 
   * @return An image rows x cols representing the heatmap of the
   * visited pixels order
   */
  cv::Mat heatmap_image(int rows, int cols, const std::vector<std::pair<int, int>>& ord) {
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (size_t i = 0, len = ord.size(); i < len; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(len);
      cv::Vec3f interpolated_color = BLUE_COLOR * (1.0f - t) + YELLOW_COLOR * t; // interpolate colors

      int r = ord[i].first;
      int c = ord[i].second;

      // Bounds check
      if (r >= 0 && r < img.rows && c >= 0 && c < img.cols) {
        cv::Vec3b& pixel = img.at<cv::Vec3b>(r, c);
        pixel[0] = cv::saturate_cast<uchar>(interpolated_color[0]); // Blue
        pixel[1] = cv::saturate_cast<uchar>(interpolated_color[1]); // Green
        pixel[2] = cv::saturate_cast<uchar>(interpolated_color[2]); // Red
      }
    }
    return img;
  }

  /**
   * @brief Draws the path of pixels followed by the Space-Filling Curve
   *
   * The image is resized by 'expand_size'. A line is then drawn between
   * consecutive pixels centers. If the image given is grayscale, converts to
   * RGB.
   *
   * @param original_img The image to draw on.
   * @param ord A vector of (row, col) pairs defining the path in the image.
   * @param expand_size The integer factor to scale the image by (e.g., 3).
   * @param color The BGR color of the path to draw.
   * 
   * @return A new image with the path followed by the Space-Filling Curve
   */
  cv::Mat path_image(const cv::Mat& original_img,
                       const std::vector<std::pair<int, int>>& ord,
                       int expand_size = 3,
                       const cv::Scalar& color = cv::Scalar(0, 0, 255)) {

    if (original_img.empty() || expand_size <= 0) {
      std::cerr << "Error: bad parameters" << std::endl;
      return original_img; // Safety check
    }
    cv::Mat img = original_img.clone();

    // if there is only one channel convert to grayscale
    if(img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    cv::Size original_size(img.cols, img.rows);
    cv::Size new_size(img.cols * expand_size, img.rows * expand_size);
    cv::resize(img, img, new_size);

    for (size_t i = 1, len = ord.size(); i < len; ++i) {
      cv::Point pa(ord[i - 1].second * expand_size + expand_size / 2,
                   ord[i - 1].first * expand_size + expand_size / 2);

      cv::Point pb(ord[i].second * expand_size + expand_size / 2,
                   ord[i].first * expand_size + expand_size / 2);

      // Draw a line between pixel centers
      cv::line(img, pa, pb, color);
    }
    // cv::resize(img, img, original_size);
    return img;
  }

  void process_image(cv::Mat& output_img, const cv::Mat& input_img, const std::vector<std::pair<int, int>>& ord, int col_index, int pixel_width) {
    for(int i = 0, len = (int)ord.size(); i < len; ++i) {
      auto [r, c] = ord[i];
      cv::Scalar pixel_color;
      
      if(input_img.channels() == 1) {
        uchar gray_value = input_img.at<uchar>(r, c);
        
        pixel_color = cv::Scalar(gray_value);
      } else {
        cv::Vec3b bgr_color = input_img.at<cv::Vec3b>(r, c);

        pixel_color = cv::Scalar(bgr_color[0], bgr_color[1], bgr_color[2]);
      }

      cv::Point p1(col_index * pixel_width, i);
      cv::Point p2((col_index + 1) * pixel_width - 1, i);
      cv::rectangle(output_img, p1, p2, pixel_color);
    }
  }

  template<typename T>
  using Vec3D = std::vector<std::vector<std::vector<T>>>;

  auto convert_8bit_image_to_vector(const cv::Mat& image) {
    Vec3D<int> converted_image;
    if (image.empty() || image.depth() != CV_8U) {
      std::cerr << "Error: Image must be 8-bit (CV_8U)." << std::endl;
      return converted_image;
    }

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    // Initialize with correct dimensions
    converted_image.assign(rows, std::vector<std::vector<int>>(cols, std::vector<int>(channels)));

    for (int r = 0; r < rows; ++r) {
      const unsigned char* row_ptr = image.ptr<unsigned char>(r);
      for (int c = 0; c < cols; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          // (c * channels) is the start of the pixel
          // + ch is the specific channel
          converted_image[r][c][ch] = static_cast<int>(row_ptr[c * channels + ch]);
        }
      }
    }
    return converted_image;
  }

  auto convert_32_float_image_to_vector(const cv::Mat& image) {
    Vec3D<double> converted_image;
    if (image.empty() || image.depth() != CV_32F) {
      std::cerr << "Error: Image must be 32-bit float (CV_32F)." << std::endl;
      return converted_image;
    }

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    // Initialize with correct dimensions
    converted_image.assign(rows, std::vector<std::vector<double>>(cols, std::vector<double>(channels)));

    for (int r = 0; r < rows; ++r) {
      const float* row_ptr = image.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          converted_image[r][c][ch] = static_cast<double>(row_ptr[c * channels + ch]);
        }
      }
    }
    return converted_image;
  }


} // namespace opencv_util

#endif // OPENCV_UTIL_HPP