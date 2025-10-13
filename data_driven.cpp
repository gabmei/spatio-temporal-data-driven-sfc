#include<iostream>
#include<vector>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

const int DIR_X[4] = {+0, +1, -0, -1};
const int DIR_Y[4] = {+1, +0, -1, -0};

void add_edge(auto& graph, std::pair<int, int> a, std::pair<int, int> b) {
  // assert((int)graph.size() < std::max(a.first, b.first));
  // assert((int)graph[a.first].size() < a.second);
  // assert((int)graph[b.first].size() < b.second);

  // if((int)graph.size() < std::max(a.first, b.first)) {
  //   std::cout << "Bad edge1: " << a.first << ' ' << a.second << " --- " << b.first << ' ' << b.second << '\n';
  //   exit(1);
  // }
  // if((int)graph[a.first].size() < a.second) {
  //   std::cout << "Bad edge2: " << a.first << ' ' << a.second << " --- " << b.first << ' ' << b.second << '\n';
  //   exit(1);
  // }
  // if((int)graph[b.first].size() < b.second) {
  //   std::cout << "Bad edge3: " << a.first << ' ' << a.second << " --- " << b.first << ' ' << b.second << '\n';
  //   exit(1);
  // }
  graph[a.first][a.second].emplace(b.first, b.second);
  graph[b.first][b.second].emplace(a.first, a.second);
}


int cross(std::pair<int, int> u, std::pair<int, int> v) {
  return u.first * v.second - u.second * v.first;
}

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
auto initial_adj(int r, int c) {
  std::vector adj(r, std::vector<std::set<std::pair<int, int>>>(c));
  for(int i = 0; i < r / 2; ++i) {
    for(int j = 0; j < c / 2; ++j) {
      auto cycle = get_node_cycle({i, j});
      for(int e = 0; e < 4; ++e) {
        int ne = e + 1 == 4 ? 0 : e + 1;
        add_edge(adj, cycle[e], cycle[ne]);
      }
    }
  }
  return adj;
}
int pixel_edge_cost(std::pair<int, int> a, std::pair<int, int> b, const cv::Mat& img) {
  auto ra = img.at<cv::Vec3b>(a.first, a.second);
  auto rb = img.at<cv::Vec3b>(b.first, b.second);
  int cost = 0;
  for(int i = 0; i < 3; ++i) {
    cost += abs(ra[i] - rb[i]);
  }
  return cost;
}

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

int adj_edge_cost(std::pair<int, int> id_a, std::pair<int, int> id_b, const cv::Mat& img) {
  int cost = 0;
  auto cycle_b = get_node_cycle(id_b);
  std::pair<int, int> dir_ab = {id_b.first - id_a.first, id_b.second - id_a.second};
  for(int e = 0; e < 4; ++e) {
    int ne = e + 1 == 4 ? 0 : e + 1;
    std::pair<int, int> dir(cycle_b[ne].first - cycle_b[e].first, cycle_b[ne].second - cycle_b[e].second);
    auto u = cross(dir_ab, dir);
    if(u <= 0) { // parallel or clockwise 
      cost += pixel_edge_cost(cycle_b[e], cycle_b[ne], img);
    }
  }
  for(auto [u, v] : get_removed_edges(id_a, id_b)) {
    cost -= pixel_edge_cost(u, v, img);
  }
  for(auto [u, v] : get_added_edges(id_a, id_b)) {
    cost += pixel_edge_cost(u, v, img);
  }
  return cost;
}


const int BLOCK = 4;
const double ALPHA = 0.1;
const std::pair<double, double> BLOCK_CENTER = {(BLOCK - 1) / 2.0, (BLOCK - 1) / 2.0};
double block_edge_cost(std::pair<int, int> id_b) {
  id_b.first %= BLOCK;
  id_b.second %= BLOCK;
  double dx = id_b.first - BLOCK_CENTER.first, dy = id_b.second - BLOCK_CENTER.second;
  return sqrt(dx * dx + dy * dy);
}
double data_driven_cost(std::pair<int, int> id_a, std::pair<int, int> id_b, const cv::Mat& img) {
  return (1 - ALPHA) * adj_edge_cost(id_a, id_b, img) + ALPHA * block_edge_cost(id_b);
}

struct DisjointSetUnion {
  std::vector<int> p;
  DisjointSetUnion(int n) : p(n, -1) {}
  int root(int a) { return p[a] < 0 ? a : p[a] = root(p[a]); }
  int size(int x) { return -p[root(x)]; }
  bool unite(int a, int b) {
    a = root(a), b = root(b);
    if(a == b) {
      return false;
    }
    if(size(a) < size(b)) {
      std::swap(a, b);
    }
    p[a] += p[b];
    p[b] = a;
    return true;
  }
};
std::vector<std::pair<int, int>> prim(const cv::Mat& img) {
  int r = img.rows, c = img.cols;
  std::vector par(r / 2, std::vector<std::pair<int, int>>(c / 2, std::make_pair(-1, -1)));
  std::cerr << r << ' ' << c << '\n';
  auto adj = initial_adj(r, c);

  using iii = std::tuple<double, int, int>;
  std::priority_queue<iii, std::vector<iii>, std::greater<iii>> pq;
  std::vector min_w(r / 2, std::vector<double>(c / 2, std::numeric_limits<int>::max()));
  std::vector is_selected(r / 2, std::vector<bool>(c / 2, false));
  int select_count = 0;

  pq.emplace(min_w[0][0] = 0, 0, 0);
  while(!pq.empty()) {
    auto [d, id_x, id_y] = pq.top();
    pq.pop();
    std::pair<int, int> id = {id_x, id_y};
    // std::cout << "on: " << id.first << ' ' << id.second << '\n';
    auto id_par = par[id_x][id_y];
    if(is_selected[id_x][id_y]) continue;
    is_selected[id_x][id_y] = true;
    select_count += 1;
    if(id_par.first != -1) {
      // not the root
      // std::cout << "edge from group " << id_par.first << ' ' << id_par.second << " to " << id.first << ' ' << id.second << '\n';
      // std::cout << "removed edges from cells\n";
      for(auto [u, v] : get_removed_edges(id_par, id)) {
        // std::cout << "rem " << u.first << ' ' << u.second << " to " << v.first << ' ' << v.second << '\n';
        adj[u.first][u.second].erase(v);
        adj[v.first][v.second].erase(u);
      }
      // std::cout << "added edges from cells " << '\n';
      for(auto [u, v] : get_added_edges(id_par, id)) {
        // std::cout << "add " << u.first << ' ' << u.second << " to " << v.first << ' ' << v.second << '\n';
        adj[u.first][u.second].emplace(v);
        adj[v.first][v.second].emplace(u);
      }
    }

    for(int i = 0; i < 4; ++i) {
      int id_nx = id_x + DIR_X[i], id_ny = id_y + DIR_Y[i];
      if(id_nx < 0 || id_ny < 0 || id_nx >= r / 2 || id_ny >= c / 2 || is_selected[id_nx][id_ny]) continue;
      auto cost = data_driven_cost({id_x, id_y}, {id_nx, id_ny}, img);
      // auto cost = 1;

      if(min_w[id_nx][id_ny] > cost) {
        pq.emplace(min_w[id_nx][id_ny] = cost, id_nx, id_ny);
        par[id_nx][id_ny] = {id_x, id_y};
      }
    }
  }

  int lo = 5, hi = 0;
  for(int x = 0; x < r; ++x) {
    for(int y = 0; y < c; ++y) {
      int sz = (int)adj[x][y].size();
      lo = std::min(lo, sz);
      hi = std::max(hi, sz);
    }
  }
  std::cerr << "min and max degree (it should be both two)" << '\n';
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

  std::cout << "number of components (it should be one): " << ncomps << '\n';

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

  std::cerr << "SELECTED " << select_count * 4 << ' ' << pixel_order.size() << '\n';
  return pixel_order;
}

const int EXPAND_SIZE = 3;
int main() {
  cv::Mat image;
  image = cv::imread("image_test2.png", 1);
  cv::imshow("Cute cat", image);
  std::vector<std::pair<int, int>> ord = prim(image);
  auto img2 = image.clone();
  cv::Size new_size(image.cols * EXPAND_SIZE, image.rows * EXPAND_SIZE);
  cv::resize(img2, img2, new_size, 1, 1);
  // size_t len = ord.size();
  // cv::Vec3f color_start(255.0f, 0.0f, 0.0f);   // Pure Blue (B, G, R)
  // cv::Vec3f color_end(0.0f, 255.0f, 255.0f); // Pure Yellow (B, G, R)

  // for(size_t i = 0, len = ord.size(); i < len; ++i) {
  //   float t = i / float(len);
  //   cv::Vec3f interpolated_color = color_start * (1.0f - t) + color_end * t;

  //   int r = ord[i].first, c = ord[i].second;

  //   cv::Vec3b& pixel = image.at<cv::Vec3b>(r, c);
  //   pixel[0] = cv::saturate_cast<uchar>(interpolated_color[0]); // Blue
  //   pixel[1] = cv::saturate_cast<uchar>(interpolated_color[1]); // Green
  //   pixel[2] = cv::saturate_cast<uchar>(interpolated_color[2]); // Red
  // }


  for(size_t i = 1, len = ord.size(); i < len; ++i) {
    cv::Point pa(ord[i - 1].second * EXPAND_SIZE + EXPAND_SIZE / 2, ord[i - 1].first * EXPAND_SIZE + EXPAND_SIZE / 2);
    cv::Point pb(ord[i].second * EXPAND_SIZE + EXPAND_SIZE / 2, ord[i].first * EXPAND_SIZE + EXPAND_SIZE / 2);

    cv::Scalar color(0, 0, 255);
    cv::line(img2, pa, pb, color);
  }
  // paint image pixels define by ord here
  cv::imshow("Context base cat", image);
  cv::imwrite("output_cat.jpg", image);
  cv::imshow("resized data_driven cat.jpg", img2);
  cv::waitKey(0);
  return 0;
}