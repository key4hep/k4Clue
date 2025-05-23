/*
 * Copyright (c) 2020-2024 Key4hep-Project.
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
#include "CLUECalorimeterHit.h"
#include <cmath>

namespace clue {

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch) : CalorimeterHit(ch) {
  setR();
  setEta();
  setPhi();
}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion,
                                       const int layer)
    : CalorimeterHit(ch), m_detectorRegion(detRegion), m_layer(layer) {
  setR();
  setEta();
  setPhi();
}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion,
                                       const int layer, const CLUECalorimeterHit::Status status, const int clusterIndex,
                                       const float rho, const float delta)
    : CalorimeterHit(ch), m_rho(rho), m_delta(delta), m_detectorRegion(detRegion), m_status(status), m_layer(layer),
      m_clusterIndex(clusterIndex) {
  setR();
  setEta();
  setPhi();
}

uint32_t CLUECalorimeterHit::getLayer() const { return m_layer; }
bool CLUECalorimeterHit::inBarrel() const { return (m_detectorRegion == barrel ? true : false); }
bool CLUECalorimeterHit::inEndcap() const { return (m_detectorRegion == endcap ? true : false); }
bool CLUECalorimeterHit::isFollower() const { return (m_status == follower ? true : false); }
bool CLUECalorimeterHit::isSeed() const { return (m_status == seed ? true : false); }
bool CLUECalorimeterHit::isOutlier() const { return (m_status == outlier ? true : false); }
float CLUECalorimeterHit::getRho() const { return m_rho; }
float CLUECalorimeterHit::getDelta() const { return m_delta; }
float CLUECalorimeterHit::getR() const { return m_r; }
float CLUECalorimeterHit::getEta() const { return m_eta; }
float CLUECalorimeterHit::getPhi() const { return m_phi; }
int32_t CLUECalorimeterHit::getClusterIndex() const { return m_clusterIndex; };

void CLUECalorimeterHit::setEta() { m_eta = -1. * log(tan(atan2(m_r, getPosition().z) / 2.)); }

void CLUECalorimeterHit::setPhi() { m_phi = atan2(getPosition().y, getPosition().x); }

void CLUECalorimeterHit::setR() {
  m_r = float(sqrt(getPosition().x * getPosition().x + getPosition().y * getPosition().y));
}

} // namespace clue

std::ostream& operator<<(std::ostream& o, const clue::CLUECalorimeterHit& value) {
  o << "Energy: " << value.getEnergy() << "\n"
    << "Time: " << value.getTime() << "\n"
    << "Position: (" << value.getPosition().x << ", " << value.getPosition().y << ", " << value.getPosition().z << ")\n"
    << "Layer: " << value.getLayer() << "\n"
    << "Region: " << (value.inBarrel() ? "Barrel" : "Endcap") << "\n"
    << "Status: "
    << (value.isSeed()       ? "Seed"
        : value.isFollower() ? "Follower"
                             : "Outlier")
    << "\n"
    << "Rho: " << value.getRho() << "\n"
    << "Delta: " << value.getDelta() << "\n"
    << "Cluster Index: " << value.getClusterIndex() << "\n";
  return o;
}
