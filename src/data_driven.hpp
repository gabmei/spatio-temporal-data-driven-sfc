#ifndef DATA_DRIVEN_H
#define DATA_DRIVEN_H
#include "distance.hpp"
#include <cmath>

/**
 * @brief Implementation class for calculating distances during prim's algorithm
 * with the Data Driven method.
 * 
 * @tparam distance_type the numerical type for distances. The data driven approach
 * needs either float or double
 * @tparam grid_type the numerical type of each color channel on each position
 * of the grid. Ideally, it should be the same as distance_type.
 */
template<typename distance_type, typename grid_type>
class DataDrivenDistance : public Distance<distance_type, grid_type> {
public:
  /**
   * @brief Constructs the Data Driven Distance calculator.
   * @param grid A const reference to the 3D grid data [x][y][channel].
   * @param ALPHA An auxiliary weight value between 0-1 for distance calculation
   * @param BLOCK The small circuits are divided into blocks of size BLOCK x BLOCK
   */
  DataDrivenDistance(const std::vector<std::vector<std::vector<grid_type>>>& grid, distance_type ALPHA, int BLOCK) : 
    Distance<distance_type, grid_type>{grid}, 
    ALPHA { ALPHA },
    BLOCK { BLOCK }                           
  {                                           
    BLOCK_CENTER = {(BLOCK - 1) / 2.0, (BLOCK - 1) / 2.0};
  }
  /**
   * @brief Function for calculating edge distances
   * @details the positions of the initial small circuits are passed as parameters.
   * During the prim's algorithm execution, id_a belong to a expanding tree, while
   * id_b is an unvisited node.
   * @param id_a The [x, y] coordinates of the node in the tree.
   * @param id_b The [x, y] coordinates of the unvisited node.
   * @return The calculated cost or distance.
   */
  distance_type get_distance(std::pair<int, int> id_a, std::pair<int, int> id_b) const override;
private:
  distance_type ALPHA;
  int BLOCK;
  std::pair<distance_type, distance_type> BLOCK_CENTER;
  distance_type adj_edge_cost(std::pair<int, int> id_a, std::pair<int, int> id_b) const;
  distance_type pixel_edge_cost(std::pair<int, int> a, std::pair<int, int> b) const;
  distance_type block_edge_cost(std::pair<int, int> id_b) const;
};


template<typename distance_type, typename grid_type>
distance_type DataDrivenDistance<distance_type, grid_type>::block_edge_cost(std::pair<int, int> id_b) const {
  id_b.first %= BLOCK;
  id_b.second %= BLOCK;
  auto dx = static_cast<distance_type>(id_b.first) - BLOCK_CENTER.first, dy = static_cast<distance_type>(id_b.second) - BLOCK_CENTER.second;
  return std::sqrt(dx * dx + dy * dy);
}

template<typename distance_type, typename grid_type>
distance_type DataDrivenDistance<distance_type, grid_type>::pixel_edge_cost(std::pair<int, int> a, std::pair<int, int> b) const {
  distance_type pixel_cost = 0;
  auto& ra = this->grid[a.first][a.second];
  auto& rb = this->grid[b.first][b.second];
  
  for(size_t i = 0, len = ra.size(); i < len; ++i) {
    pixel_cost += std::abs(static_cast<distance_type>(ra[i]) - static_cast<distance_type>(rb[i]));
  }
  return pixel_cost;
}

template<typename distance_type, typename grid_type>
distance_type DataDrivenDistance<distance_type, grid_type>::adj_edge_cost(std::pair<int, int> id_a, std::pair<int, int> id_b) const {
  distance_type cost = 0;
  auto cycle_b = util::get_node_cycle(id_b);
  std::pair<int, int> dir_ab = {id_b.first - id_a.first, id_b.second - id_a.second};
  for(int e = 0; e < 4; ++e) {
    int ne = e + 1 == 4 ? 0 : e + 1;
    std::pair<int, int> dir(cycle_b[ne].first - cycle_b[e].first, cycle_b[ne].second - cycle_b[e].second);
    auto u = util::cross(dir_ab, dir);
    if(u <= 0) { // parallel or clockwise 
      cost += pixel_edge_cost(cycle_b[e], cycle_b[ne]);
    }
  }
  for(auto [u, v] : util::get_removed_edges(id_a, id_b)) {
    cost -= pixel_edge_cost(u, v);
  }
  for(auto [u, v] : util::get_added_edges(id_a, id_b)) {
    cost += pixel_edge_cost(u, v);
  }
  return cost;
}

template<typename distance_type, typename grid_type>
distance_type DataDrivenDistance<distance_type, grid_type>::get_distance(std::pair<int, int> id_a, std::pair<int, int> id_b) const {
  return (1 - ALPHA) * adj_edge_cost(id_a, id_b) + ALPHA * block_edge_cost(id_b);
}

#endif // !DATA_DRIVEN_H