#ifndef DSU_HPP
#define DSU_HPP

#include <vector>
#include <utility> // For std::swap

/**
 * @brief Implements the Disjoint Set Union (DSU) data structure.
 * Also known as Union-Find.
 */
struct DisjointSetUnion { 
  std::vector<int> p; 
  
  DisjointSetUnion(int n) : p(n, -1) {} 
  
  int root(int a) { 
    return p[a] < 0 ? a : p[a] = root(p[a]); 
  } 
  
  int size(int x) { 
    return -p[root(x)]; 
  } 
  
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

#endif // DSU_HPP