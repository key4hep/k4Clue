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
#ifndef K4CLUE_CLUECALORIMETERHIT_H
#define K4CLUE_CLUECALORIMETERHIT_H

#include "edm4hep/CalorimeterHit.h"
#include <GaudiKernel/DataObject.h>

namespace clue {

class CLUECalorimeterHit : public edm4hep::CalorimeterHit, public DataObject {
public:
  using CalorimeterHit::CalorimeterHit;

  enum Status { outlier = 0, follower, seed };

  enum DetectorRegion { barrel = 0, endcap };

  /// constructors
  CLUECalorimeterHit(const CalorimeterHit& ch);

  CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion, const int layer);

  CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion, const int layer,
                     const CLUECalorimeterHit::Status status, const int clusterIndex, const float rho,
                     const float delta);

  /// Access the layer number
  uint32_t getLayer() const;

  /// Access the region of calorimeter
  bool inBarrel() const;

  /// Access the region of calorimeter
  bool inEndcap() const;

  /// Status follower value
  bool isFollower() const;

  /// Status outlier value
  bool isOutlier() const;

  /// Status seed value
  bool isSeed() const;

  /// Access the delta
  float getDelta() const;

  /// Access the rho
  float getRho() const;

  /// Access the transverse position
  float getR() const;

  /// Access the eta
  float getEta() const;

  /// Access the phi
  float getPhi() const;

  /// Access cluster index
  int32_t getClusterIndex() const;

  /// Set hit transverse global position, pseudorapidity and phi
  void setR();
  void setEta();
  void setPhi();

  void setRho(float rho) { m_rho = rho; }
  void setDelta(float delta) { m_delta = delta; }
  void setStatus(Status status) { m_status = status; }
  void setClusterIndex(int32_t clIdx) { m_clusterIndex = clIdx; }

private:
  float m_eta;
  float m_phi;
  float m_r;
  float m_rho;
  float m_delta;
  uint8_t m_detectorRegion;
  uint8_t m_status;
  uint32_t m_layer;
  int32_t m_clusterIndex;
};

class CLUECalorimeterHitCollection : public DataObject {
public:
  std::vector<CLUECalorimeterHit> vect;
};

} // namespace clue

std::ostream& operator<<(std::ostream& o, const clue::CLUECalorimeterHit& value);

#endif
