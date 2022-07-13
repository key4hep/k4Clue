#ifndef constexpr_cmath_h
#define constexpr_cmath_h

#include <cstdint>

namespace reco {
  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num)
                                                                  : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }

  // reduce to [-pi,pi]
  template <typename T>
  constexpr T reduceRange(T x) {
    constexpr T o2pi = 1. / (2. * M_PI);
    if (std::abs(x) <= T(M_PI))
      return x;
    T n = std::round(x * o2pi);
    return x - n * T(2. * M_PI);
  }

  // return a value of phi into interval [-pi,+pi]
  template <typename T>
  constexpr T normalizedPhi(T phi) {
    return reduceRange(phi);
  }

  template <typename T>
  constexpr T deltaPhi(T phi1, T phi2) {
    return reduceRange(phi1 - phi2);
  }

};  // namespace reco

#endif
