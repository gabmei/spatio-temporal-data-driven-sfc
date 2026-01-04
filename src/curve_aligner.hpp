#ifndef CURVE_ALIGNER_H
#define CURVE_ALIGNER_H

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <string>
#include <format>
#include <stdexcept>

#include "convolutions.hpp"

/**
 * @namespace curve_aligner
 * @brief A namespace containing methods for reordering multiple frames of space-filling curves paths
 * 
 * Reorder consecutive frames by minimizing pixel difference
 */
namespace curve_aligner {

struct AlignmentResult {
  double score;
  int shift;
  bool should_maximize; // true if higher is better (Correlation), false if lower is better (L1)

  bool is_better_than(const AlignmentResult& other) const {
    if (should_maximize) return score > other.score;
    return score < other.score;
  }
};  

template<typename T>
std::vector<std::vector<double>> linearize_image(const std::vector<std::vector<std::vector<T>>>& image, const std::vector<std::pair<int, int>>& path) {
  std::vector<std::vector<double>> linearized_image;
  linearized_image.reserve(path.size());
  for(auto [r, c] : path) {
    std::vector<double> pixel_info;
    for(T x : image[r][c]) {
      // for all purposes, it's safer to use floating arithmetics from now on
      pixel_info.emplace_back(static_cast<double>(x));
    }
    linearized_image.emplace_back(pixel_info);
  }
  return linearized_image;
}

double calculate_pixel_difference(const std::vector<double>& u, const std::vector<double>& v) {
  double cost = 0;
  for(size_t i = 0, len = u.size(); i < len; ++i) {
    cost += std::abs(u[i] - v[i]);
  }
  return cost;
}

double calculate_pixel_weight(const auto& current_path, const auto& previous_path, size_t rot) {
  double score = 0;
  for(size_t lst = 0, cur = rot, len = current_path.size(); lst < len; ++lst, cur = (cur + 1 < len ? cur + 1 : 0)) {
    score += calculate_pixel_difference(current_path[cur], previous_path[lst]);
  }
  return score;
}

AlignmentResult run_l1_norm_strategy(const auto& current_path, const auto& previous_path) {
  double best_rotation_score = std::numeric_limits<double>::max();
  int best_rotation_id = -1;

  for(size_t rot = 0, sz = current_path.size(); rot < sz; ++rot) {
    double rotation_score = calculate_pixel_weight(current_path, previous_path, rot);
    if(rotation_score < best_rotation_score) {
      best_rotation_score = rotation_score;
      best_rotation_id = static_cast<int>(rot);
    }
  }
  return {best_rotation_score, best_rotation_id, false};
}

AlignmentResult run_l2_norm_strategy(auto current_path, const auto& previous_path) {
  size_t N = current_path.size();
  for(size_t i = 0; i < N; ++i) {
    current_path.emplace_back(current_path[i]);
  }

  std::vector<double> total_correlation(N, 0.0);
  auto copy_from_channel = [](const auto& path, size_t c) {
    size_t sz = path.size();
    std::vector<double> a(sz);
    for(size_t i = 0; i < sz; ++i) {
      a[i] = path[i][c];
    }
    return a;
  };
  for(size_t c = 0, channels = current_path[0].size(); c < channels; ++c) {
    auto a = copy_from_channel(current_path, c);
    auto b = copy_from_channel(previous_path, c);

    auto correlation = convolutions::correlate_valid(a, b);
    
    for(size_t i = 0; i < N; ++i) {
      total_correlation[i] += correlation[i];
    }
  }
  auto best_rotation_score = std::max_element(begin(total_correlation), end(total_correlation));
  int best_rotation_id = int(best_rotation_score - begin(total_correlation));

  return {*best_rotation_score, best_rotation_id, true};
}

AlignmentResult calculate_best_rotation(auto current_path, const auto& previous_path, const std::string& align_strategy, bool try_reverse = false) {
  if(try_reverse) {
    reverse(begin(current_path), end(current_path));
  }

  if(align_strategy == "L1-norm") {
    return run_l1_norm_strategy(current_path, previous_path);
  }
  if(align_strategy == "L2-norm") {
    return run_l2_norm_strategy(current_path, previous_path);
  }
  throw std::runtime_error(
    std::format("Unsuported alignment strategy found = {}", align_strategy)
  );
}

template<typename T>
void reorder_frames(const std::vector<std::vector<std::vector<std::vector<T>>>>& all_images, std::vector<std::vector<std::pair<int, int>>>& all_paths, const std::string& align_strategy) {
  if(align_strategy == "None") {
    return;
  }
  auto previous_path = linearize_image(all_images[0], all_paths[0]);
  for(size_t i = 1, len = all_paths.size(); i < len; ++i) {
    auto current_path = linearize_image(all_images[i], all_paths[i]);

    auto rot_result = calculate_best_rotation(current_path, previous_path, align_strategy);
    auto rev_rot_result = calculate_best_rotation(current_path, previous_path, align_strategy, true);

    if(rev_rot_result.is_better_than(rot_result)) {
      reverse(begin(all_paths[i]), end(all_paths[i]));
      std::rotate(begin(all_paths[i]), begin(all_paths[i]) + rev_rot_result.shift, end(all_paths[i]));

      reverse(begin(current_path), end(current_path));
      std::rotate(begin(current_path), begin(current_path) + rev_rot_result.shift, end(current_path));
    } else {
      std::rotate(begin(all_paths[i]), begin(all_paths[i]) + rot_result.shift, end(all_paths[i]));

      std::rotate(begin(current_path), begin(current_path) + rot_result.shift, end(current_path));
    }

    previous_path.swap(current_path);
  }
}

}

#endif // !CURVE_ALIGNER_H