#ifndef DISTANCE_H
#define DISTANCE_H
#include "util.hpp"
/**
 * @brief Base class for calculating distances during prim's algorithm.
 * 
 * @tparam distance_type the numerical type for distances (e.g., int, float, double)
 * @tparam grid_type the numerical type of each color channel on each position
 * of the grid. Ideally, it should be the same as distance_type.
 */
template<typename distance_type, typename grid_type>
class Distance {
public:
  /**
   * @brief Constructs the Distance calculator.
   * @param grid A const reference to the 3D grid data [x][y][channel].
   */
  Distance(const std::vector<std::vector<std::vector<grid_type>>>& grid) : grid { grid } {};
  /**
   * @brief Pure virtual function for calculating edge distances
   * @details the positions of the initial small circuits are passed as parameters.
   * During the prim's algorithm execution, id_a belong to a expanding tree, while
   * id_b is an unvisited node.
   * @param id_a The [x, y] coordinates of the node in the tree.
   * @param id_b The [x, y] coordinates of the unvisited node.
   * @return The calculated cost or distance.
   */
  virtual distance_type get_distance(std::pair<int, int> id_a, std::pair<int, int> id_b) const = 0;
protected:
  std::vector<std::vector<std::vector<grid_type>>> grid;
};

#endif // !DISTANCE_H