#ifndef CONVOLUTIONS_H
#define CONVOLUTIONS_H

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <string>
#include <complex>
/**
 * @namespace convolutions
 * @brief A namespace containing methods related to fft convolutions
 * 
 * These methods are used by the curve_aligner.hpp class for the l2-norm strategy
 */
namespace convolutions {

// Special thanks to https://github.com/kth-competitive-programming/kactl/blob/main/content/numerical/FastFourierTransform.h
using Complex = std::complex<double>;
void fft(std::vector<Complex>& a) {
  std::vector<std::complex<long double>> R(2, 1); 
  std::vector<Complex> rt(2, 1); // (^ 10% faster if double)
  int n = (int)a.size(), L = 31 - __builtin_clz(n);
  for(int k = 2; k < n; k *= 2) {
    R.resize(n);
    rt.resize(n);
    auto x = std::polar(1.0L, std::acos(-1.0L) / k);
    for (int i = k; i < 2 * k; ++i) {
      rt[i] = R[i] = i & 1 ? R[i / 2] * x : R[i / 2];
    }
  }
  std::vector<int> rev(n);
  for(int i = 0; i < n; ++i) {
    rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
    if (i < rev[i]) swap(a[i], a[rev[i]]);
  }
  for (int k = 1; k < n; k *= 2)
    for (int i = 0; i < n; i += 2 * k)
      for (int j = 0; j < k; ++j) {
        // Complex z = rt[j+k] * a[i+j+k]; // (25% faster if hand-rolled)  /// include-line
        auto x = (double *)&rt[j+k], y = (double *)&a[i+j+k];              /// exclude-line
        Complex z(x[0]*y[0] - x[1]*y[1], x[0]*y[1] + x[1]*y[0]);           /// exclude-line
        a[i + j + k] = a[i + j] - z;
        a[i + j] += z;
      }
}

std::vector<double> convolution(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.empty() || b.empty()) return {};
  std::vector<double> res((int)a.size() + (int)b.size() - 1);
  int L = 32 - __builtin_clz((int)res.size()), n = 1 << L;
  std::vector<Complex> in(n), out(n);
  copy(a.begin(), a.end(), in.begin());
  for (int i = 0; i < (int)b.size(); ++i) in[i].imag(b[i]);
  fft(in);
  for (Complex& x: in)  x *= x;
  for (int i = 0; i < n; ++i) out[i] = in[-i & (n - 1)] - conj(in[i]);
  fft(out);
  for (int i = 0; i < (int)res.size(); ++i) {
    res[i] = imag(out[i]) / (4 * n);
  }
  return res;
}

std::vector<double> correlate_valid(const std::vector<double>& a, std::vector<double> b) {
  if (a.empty() || b.empty()) return {};
  int sz_a = (int)a.size(), sz_b = (int)b.size();
  if(sz_a < sz_b) return {};
  reverse(begin(b), end(b));
  auto full = convolution(a, b);
  return std::vector<double>(begin(full) + sz_b - 1, begin(full) + sz_a);
}

}

#endif // !CONVOLUTIONS_H