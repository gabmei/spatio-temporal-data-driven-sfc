#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <utility> // For std::pair
#include <set>     // For std::set

/**
 * @namespace util
 * @brief A namespace for common operations used on this Space-Filling Curve application.
 */

namespace util {

inline constexpr int DIR_X[4] = {+0, +1, -0, -1};
inline constexpr int DIR_Y[4] = {+1, +0, -1, -0};

/**
 * @brief Calculates the 2D cross product of two vectors (pairs).
 * 
 * In this application, it's a utility function used to calculate the cross
 * between unit vectors.
 */
int cross(std::pair<int, int> u, std::pair<int, int> v) {
  return u.first * v.second - u.second * v.first;
}

/**
 * @brief Gets the 4 corner coordinates for a node ID.
 */
std::vector<std::pair<int, int>> get_node_cycle(std::pair<int, int> id) {
  int x = id.first * 2, y = id.second * 2;
  std::vector<std::pair<int, int>> cycle;
  for(int i = 0; i < 4; ++i) {
    cycle.emplace_back(x, y);
    x += DIR_X[i];
    y += DIR_Y[i];
  }
  return cycle;
}

/**
 * @brief Calculates edges to be removed when merging two nodes.
 */
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> get_removed_edges(std::pair<int, int> id_a, std::pair<int, int> id_b) {
  auto cycle_b = get_node_cycle(id_b);
  std::pair<int, int> dir_ab = {id_b.first - id_a.first, id_b.second - id_a.second};
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> rem;
  for(int e = 0; e < 4; ++e) {
    int ne = e + 1 == 4 ? 0 : e + 1;
    std::pair<int, int> dir(cycle_b[ne].first - cycle_b[e].first, cycle_b[ne].second - cycle_b[e].second);
    auto u = cross(dir_ab, dir);

    if(u <= 0) { // parallel or clockwise 
      continue;
    } 
    // counterclockwise
    rem.emplace_back(cycle_b[e], cycle_b[ne]);
  }

  auto cycle_a = get_node_cycle(id_a);
  for(int e = 0; e < 4; ++e) {
    int ne = e + 1 == 4 ? 0 : e + 1;
    std::pair<int, int> dir(cycle_a[ne].first - cycle_a[e].first, cycle_a[ne].second - cycle_a[e].second);
    auto u = cross(dir_ab, dir);
    if(u == -1) { // clockwise
      rem.emplace_back(cycle_a[e], cycle_a[ne]);
    }
  }
  return rem;
}

/**
 * @brief Calculates edges to be added when merging two nodes.
 */
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> get_added_edges(std::pair<int, int> id_a, std::pair<int, int> id_b) {
  std::pair<int, int> dir_ab = {id_b.first - id_a.first, id_b.second - id_a.second};
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> add;
  std::set<std::pair<int, int>> cycle_b;
  for(auto x : get_node_cycle(id_b)) {
    cycle_b.insert(x);
  }
  auto cycle_a = get_node_cycle(id_a);
  for(auto u : cycle_a) {
    auto v = u;
    v.first += dir_ab.first;
    v.second += dir_ab.second;

    if(cycle_b.count(v)) { // from one group to another
      add.emplace_back(u, v);
    }
  }
  return add;
}

} // namespace util

#endif // UTIL_HPP