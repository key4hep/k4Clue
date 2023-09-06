/*
 * Copyright (c) 2020-2023 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
