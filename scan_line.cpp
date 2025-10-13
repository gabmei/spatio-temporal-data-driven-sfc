#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main() {
  cv::Mat image;
  image = cv::imread("So_happy_smiling_cat.jpg", 1);
  cv::imshow("Cute cat", image);

  for(int i = 0; i < image.rows; i += 2) {
    auto row = image.row(i);
    cv::flip(row, row, 1);
  }
  cv::imshow("Scan-line cute cat", image);
  cv::imwrite("scan_line_cat.jpg", image);
  cv::waitKey(0);
  return 0;
}