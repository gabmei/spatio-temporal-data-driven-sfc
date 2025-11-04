#ifndef PRIM_HPP
#define PRIM_HPP

#include <vector>
#include <set>
#include <queue>        // For std::priority_queue
#include <tuple>        // For std::tuple
#include <utility>      // For std::pair, std::make_pair
#include <limits>       // For std::numeric_limits
#include <functional>   // For std::greater
#include <iostream>     // For std::cerr

// Custom headers
#include "dsu.hpp"
#include "util.hpp"
#include "distance.hpp"

/**
 * @brief A class to run Prim's algorithm on a grid of nodes,
 * modifying an underlying pixel graph to create a space-filling curve.
 *
 * @tparam distance_type The numerical type for distances (e.g., float, double).
 * @tparam grid_type The numerical type of the grid data (e.g., int, float).
 * Ideally, it should be the same as distance_type.
 */
template<typename distance_type, typename grid_type>
class Prim {
public:
  /**
   * @brief Constructs the Prim algorithm runner.
   * @param r The number of rows in the pixel grid.
   * @param c The number of columns in the pixel grid.
   */
  Prim(int r, int c) :
    r { r }, 
    c { c }, 
    node_r { r / 2 }, 
    node_c { c / 2 },
    adj(r, std::vector<std::set<std::pair<int, int>>>(c))
  {
    std::cerr << "Grid dimensions: " << r << ' ' << c << '\n';
    initial_adj(); // Build the initial graph
  }

  /**
   * @brief Runs Prim's algorithm to generate the space-filling curve.
   * @details It uses (0, 0) as a default start position for the prim's algorithm
   *
   * @param dist_calc A reference to a Distance object (e.g., DataDrivenDistance)
   * used to calculate costs between nodes.
   * @return A vector of pixel coordinates representing the space-filling curve.
   */
  std::vector<std::pair<int, int>> run(Distance<distance_type, grid_type>& dist_calc);

private:
  int r, c;           // Pixel grid dimensions
  int node_r, node_c; // Node grid dimensions
  std::vector<std::vector<std::set<std::pair<int, int>>>> adj; // Pixel adjacency graph

  /**
   * @brief Helper to add an undirected edge to the member 'adj' graph.
   */
  void add_edge(std::pair<int, int> a, std::pair<int, int> b);

  /**
   * @brief Creates the initial pixel adjacency list for all small circuits.
   */
  void initial_adj();
};

template<typename distance_type, typename grid_type>
std::vector<std::pair<int, int>> Prim<distance_type, grid_type>::run(Distance<distance_type, grid_type>& dist_calc) {
  std::vector par(node_r, std::vector<std::pair<int, int>>(node_c, std::make_pair(-1, -1)));
  
  using iii = std::tuple<distance_type, int, int>;
  std::priority_queue<iii, std::vector<iii>, std::greater<iii>> pq;
  std::vector min_w(node_r, std::vector<distance_type>(node_c, std::numeric_limits<distance_type>::max()));
  std::vector is_selected(node_r, std::vector<bool>(node_c, false));
  int select_count = 0;

  pq.emplace(min_w[0][0] = 0, 0, 0);
  while(!pq.empty()) {
    auto [d, id_x, id_y] = pq.top();
    pq.pop();
    std::pair<int, int> id = {id_x, id_y};
    
    if(is_selected[id_x][id_y]) continue;
    is_selected[id_x][id_y] = true;
    select_count += 1;
    
    auto id_par = par[id_x][id_y];
    if(id_par.first != -1) {
      // Not the root, join it to its parent
      for(auto [u, v] : util::get_removed_edges(id_par, id)) {
        adj[u.first][u.second].erase(v);
        adj[v.first][v.second].erase(u);
      }
      for(auto [u, v] : util::get_added_edges(id_par, id)) {
        add_edge(u, v);
      }
    }

    for(int i = 0; i < 4; ++i) {
      int id_nx = id_x + util::DIR_X[i], id_ny = id_y + util::DIR_Y[i];
      if(id_nx < 0 || id_ny < 0 || id_nx >= node_r || id_ny >= node_c || is_selected[id_nx][id_ny]) continue;
      
      auto cost = dist_calc.get_distance({id_x, id_y}, {id_nx, id_ny});

      if(min_w[id_nx][id_ny] > cost) {
        pq.emplace(min_w[id_nx][id_ny] = cost, id_nx, id_ny);
        par[id_nx][id_ny] = {id_x, id_y};
      }
    }
  }

  // Debug logic to see if it generated an actual space-filling curve

  int lo = 5, hi = 0;
  for(int x = 0; x < r; ++x) {
    for(int y = 0; y < c; ++y) {
      int sz = (int)adj[x][y].size();
      lo = std::min(lo, sz);
      hi = std::max(hi, sz);
    }
  }
  std::cerr << "min and max degree (it should be both two): ";
  std::cerr << lo << ' ' << hi << '\n';

  DisjointSetUnion dsu(r * c);
  int ncomps = r * c;
  for(int x = 0; x < r; ++x) {
    for(int y = 0; y < c; ++y) {
      int a = x * c + y;
      for(auto z : adj[x][y]) {
        int nx = z.first, ny = z.second;
        int b = nx * c + ny;
        if(dsu.unite(a, b)) {
          ncomps -= 1;
        }
      }
    }
  }

  std::cerr << "number of components (it should be one): " << ncomps << '\n';

  std::vector<std::pair<int, int>> pixel_order;
  std::pair<int, int> cur = {0, 0};
  std::vector is_visited(r, std::vector<bool>(c, false));
  
  do {
    is_visited[cur.first][cur.second] = true;
    pixel_order.emplace_back(cur);
    for(auto nxt : adj[cur.first][cur.second]) {
      if(is_visited[nxt.first][nxt.second]) continue;
      cur = nxt;
      break;
    }
  } while(!is_visited[cur.first][cur.second]);

  std::cerr << "SELECTED " << select_count * 4 << " PIXELS. PATH LENGTH: " << pixel_order.size() << '\n';
  return pixel_order;
}

template<typename distance_type, typename grid_type>
void Prim<distance_type, grid_type>::add_edge(std::pair<int, int> a, std::pair<int, int> b) {
  this->adj[a.first][a.second].emplace(b);
  this->adj[b.first][b.second].emplace(a);
}

template<typename distance_type, typename grid_type>
void Prim<distance_type, grid_type>::initial_adj() {
  for(int i = 0; i < node_r; ++i) {
    for(int j = 0; j < node_c; ++j) {
      auto cycle = util::get_node_cycle({i, j}); 
      for(int e = 0; e < 4; ++e) {
        int ne = e + 1 == 4 ? 0 : e + 1;
        add_edge(cycle[e], cycle[ne]);
      }
    }
  }
}

#endif // PRIM_HPP