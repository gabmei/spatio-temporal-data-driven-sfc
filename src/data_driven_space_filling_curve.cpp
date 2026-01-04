#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>

#include "data_driven.hpp"
#include "prim.hpp"
#include "curve_aligner.hpp"

namespace py = pybind11;

struct PerformanceMetrics {
  double core_algo_time_ms;
  double total_cpp_time_ms;
};

template<typename T>
std::vector<std::vector<std::vector<T>>> reshape_image(const py::array_t<T>& input_array) {
  auto buf = input_array.request();
  int height = buf.shape[0];
  int width = buf.shape[1];

  // Check if we have a 3rd dimension (channels). If not, assume 1 channel.
  int channels = (buf.ndim == 3) ? buf.shape[2] : 1;

  std::vector<std::vector<std::vector<T>>> img(height);

  for(int r = 0; r < height; ++r) {
    img[r].resize(width);
    for(int c = 0; c < width; ++c) {
      img[r][c].resize(channels);
      for(int k = 0; k < channels; ++k) {
        if (buf.ndim == 3) {
          img[r][c][k] = input_array.at(r, c, k);
        } else {
          img[r][c][k] = input_array.at(r, c);
        }
      }
    }
  }
  return img;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> reshape_image_frame(const py::array_t<T>& input_array, int frame_idx) {
  auto buf = input_array.request();
  int height = buf.shape[1];
  int width = buf.shape[2];

  // Check if we have a 4rd dimension (channels). If not, assume 1 channel.
  int channels = (buf.ndim == 4) ? buf.shape[3] : 1;

  std::vector<std::vector<std::vector<T>>> img(height);

  for(int r = 0; r < height; ++r) {
    img[r].resize(width);
    for(int c = 0; c < width; ++c) {
      img[r][c].resize(channels);
      for(int k = 0; k < channels; ++k) {
        if (buf.ndim == 4) {
          img[r][c][k] = input_array.at(frame_idx, r, c, k);
        } else {
          img[r][c][k] = input_array.at(frame_idx, r, c);
        }
      }
    }
  }
  return img;
}

/**
 * Process a single image.
 * 
 * @param ALPHA is a weight between 0 and 1 to balance between pixel and
 * spacial relevance when creating the space-filling curve. If 0.0, only
 * pixel adjacency is considered.
 */
template<typename T>
std::pair<std::vector<std::pair<int, int>>, PerformanceMetrics> data_driven_process_image(py::array_t<T> input_array, double ALPHA, int BLOCK_SIZE) {
  auto start_total = std::chrono::steady_clock::now();
  if(input_array.ndim() != 2 && input_array.ndim() != 3) {
    throw std::runtime_error("Input image must be 2D [H,W] or 3D [H,W,C]");
  }
  // Part of pybind ovearhead
  int height = input_array.shape(0);
  int width = input_array.shape(1);
  auto img = reshape_image(input_array);

  auto start_core = std::chrono::steady_clock::now();

  // Core algorithm logic
  auto result_path = Prim<double, T>(height, width).run(DataDrivenDistance<double, T>(img, ALPHA, BLOCK_SIZE));

  auto end_time = std::chrono::steady_clock::now();  

  PerformanceMetrics stats{
    std::chrono::duration<double, std::milli>(end_time - start_core).count(),
    std::chrono::duration<double, std::milli>(end_time - start_total).count()
  };
  return {result_path, stats};
}

/**
 * Process a list of images.
 * 
 * If the align_strategy is set, it will try each cyclic shift
 * in a way that best match previous path data. Cyclic shifts
 * are considered in both directions.
 */
template<typename T>
std::pair<std::vector<std::vector<std::pair<int, int>>>, PerformanceMetrics> data_driven_process_multiple_images(py::array_t<T> input_array, double ALPHA, int BLOCK_SIZE, const std::string& align_strategy) {
  auto start_total = std::chrono::steady_clock::now();
  if(input_array.ndim() != 3 && input_array.ndim() != 4) {
    throw std::runtime_error("Input animation must be 3D [F,H,W] or 4D [F,H,W,C]");
  }
  // Part of pybind ovearhead
  int frames = input_array.shape(0);
  int height = input_array.shape(1);
  int width = input_array.shape(2);

  std::vector<std::vector<std::vector<std::vector<T>>>> all_images(frames);
  std::vector<std::vector<std::pair<int, int>>> all_paths(frames);

  for(int f = 0; f < frames; ++f) {
    all_images[f] = reshape_image_frame(input_array, f);
  }

  auto start_core = std::chrono::steady_clock::now();

  for(int f = 0; f < frames; ++f) {
    all_paths[f] = Prim<double, T>(height, width).run(DataDrivenDistance<double, T>(all_images[f], ALPHA, BLOCK_SIZE));
  }
  
  curve_aligner::reorder_frames(all_images, all_paths, align_strategy);
  auto end_time = std::chrono::steady_clock::now();

  PerformanceMetrics stats{
    std::chrono::duration<double, std::milli>(end_time - start_core).count(),
    std::chrono::duration<double, std::milli>(end_time - start_total).count()
  };
  return {all_paths, stats};
}


std::pair<std::vector<std::pair<int, int>>, PerformanceMetrics> dispatcher_benchmarked(py::array input, double ALPHA, int BLOCK_SIZE) {
  // Check the data type (dtype) of the numpy array
  if (py::isinstance<py::array_t<uint8_t>>(input)) {
    return data_driven_process_image<uint8_t>(input, ALPHA, BLOCK_SIZE);
  }
  if (py::isinstance<py::array_t<uint16_t>>(input)) {
    return data_driven_process_image<uint16_t>(input, ALPHA, BLOCK_SIZE);
  }
  if (py::isinstance<py::array_t<float>>(input)) {
    return data_driven_process_image<float>(input, ALPHA, BLOCK_SIZE);
  }
  if (py::isinstance<py::array_t<double>>(input)) {
    return data_driven_process_image<double>(input, ALPHA, BLOCK_SIZE);
  }
  throw std::runtime_error("Unsupported data type! Please provide uint8, float32, or float64.");
}

std::pair<std::vector<std::vector<std::pair<int, int>>>, PerformanceMetrics> dispatcher_animation_benchmarked(py::array input, double ALPHA, int BLOCK_SIZE, const std::string& align_strategy) {
  // Check the data type (dtype) of the numpy array
  if (py::isinstance<py::array_t<uint8_t>>(input)) {
    return data_driven_process_multiple_images<uint8_t>(input, ALPHA, BLOCK_SIZE, align_strategy);
  }
  if (py::isinstance<py::array_t<uint16_t>>(input)) {
    return data_driven_process_multiple_images<uint16_t>(input, ALPHA, BLOCK_SIZE, align_strategy);
  }
  if (py::isinstance<py::array_t<float>>(input)) {
    return data_driven_process_multiple_images<float>(input, ALPHA, BLOCK_SIZE, align_strategy);
  }
  if (py::isinstance<py::array_t<double>>(input)) {
    return data_driven_process_multiple_images<double>(input, ALPHA, BLOCK_SIZE, align_strategy);
  }
  throw std::runtime_error("Unsupported data type! Please provide uint8, float32, or float64.");
}

std::vector<std::pair<int, int>> dispatcher(py::array input, double ALPHA, int BLOCK_SIZE) {
  // Check the data type (dtype) of the numpy array
  if (py::isinstance<py::array_t<uint8_t>>(input)) {
    return data_driven_process_image<uint8_t>(input, ALPHA, BLOCK_SIZE).first;
  }
  if (py::isinstance<py::array_t<uint16_t>>(input)) {
    return data_driven_process_image<uint16_t>(input, ALPHA, BLOCK_SIZE).first;
  }
  if (py::isinstance<py::array_t<float>>(input)) {
    return data_driven_process_image<float>(input, ALPHA, BLOCK_SIZE).first;
  }
  if (py::isinstance<py::array_t<double>>(input)) {
    return data_driven_process_image<double>(input, ALPHA, BLOCK_SIZE).first;
  }
  throw std::runtime_error("Unsupported data type! Please provide uint8, float32, or float64.");
}


/**
 * Dispacher function exposed to python
 */
std::vector<std::vector<std::pair<int, int>>> dispatcher_animation(py::array input, double ALPHA, int BLOCK_SIZE, const std::string& align_strategy) {
  // Check the data type (dtype) of the numpy array
  if (py::isinstance<py::array_t<uint8_t>>(input)) {
    return data_driven_process_multiple_images<uint8_t>(input, ALPHA, BLOCK_SIZE, align_strategy).first;
  }
  if (py::isinstance<py::array_t<uint16_t>>(input)) {
    return data_driven_process_multiple_images<uint16_t>(input, ALPHA, BLOCK_SIZE, align_strategy).first;
  }
  if (py::isinstance<py::array_t<float>>(input)) {
    return data_driven_process_multiple_images<float>(input, ALPHA, BLOCK_SIZE, align_strategy).first;
  }
  if (py::isinstance<py::array_t<double>>(input)) {
    return data_driven_process_multiple_images<double>(input, ALPHA, BLOCK_SIZE, align_strategy).first;
  }
  throw std::runtime_error("Unsupported data type! Please provide uint8, float32, or float64.");
}


/**
 * Binding to python module
 */
PYBIND11_MODULE(data_driven_module, m) {
    m.doc() = "Data Driven Traversal Module";

    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
      .def_readonly("core_algo_time_ms", &PerformanceMetrics::core_algo_time_ms)
      .def_readonly("total_cpp_time_ms", &PerformanceMetrics::total_cpp_time_ms);
    
    // Exposed python function names
    m.def("get_image_traversal_path", &dispatcher, "Calculate traversal path for generic arrays");
    m.def("get_multiple_images_traversal_path", &dispatcher_animation,
      "Calculate traversal path for multiple generic arrays",
      py::arg("input"),
      py::arg("ALPHA"),
      py::arg("BLOCK_SIZE"),
      py::arg("align_strategy") = "None");

    m.def("get_image_traversal_path_benchmarked", &dispatcher_benchmarked, "Calculate traversal path for generic arrays with benchmarks");
    m.def("get_multiple_images_traversal_path_benchmarked", &dispatcher_animation_benchmarked,
      "Calculate traversal path for multiple generic arrays with benchmarks", 
      py::arg("input"),
      py::arg("ALPHA"),
      py::arg("BLOCK_SIZE"),
      py::arg("align_strategy") = "None");
}