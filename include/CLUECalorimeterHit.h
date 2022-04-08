#ifndef K4CLUE_CLUECALORIMETERHIT_H
#define K4CLUE_CLUECALORIMETERHIT_H

#include "edm4hep/CalorimeterHit.h"
#include <GaudiKernel/DataObject.h>

using namespace edm4hep;

namespace clue {

class CLUECalorimeterHit : public CalorimeterHit, public DataObject {
public:
  using CalorimeterHit::CalorimeterHit;

  enum Status { outlier = 0, follower, seed };

  enum DetectorRegion { barrel = 0, endcap };

  /// constructors
  CLUECalorimeterHit(const CalorimeterHit& ch);

  CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const CLUECalorimeterHit::DetectorRegion barrel);

  CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const CLUECalorimeterHit::DetectorRegion barrel, 
                     const CLUECalorimeterHit::Status status, const int clusterIndex, const float rho, const float delta);

  /// Access the layer number
  const std::uint64_t& getLayer() const;

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
  const float& getDelta() const;

  /// Access the rho
  const float& getRho() const;

  /// Access the transverse position
  const float& getR() const;

  /// Access the eta
  const float& getEta() const;

  /// Access the phi
  const float& getPhi() const;

  /// Set hit transverse global position, pseudorapidity and phi
  void setR();
  void setEta();
  void setPhi();

  void setRho( float rho ) { m_rho = rho; };
  void setDelta( float delta ) { m_delta = delta; };
  void setStatus( Status status ) { m_status = status; };
  void setClusterIndex( int clIdx ) { m_clusterIndex = clIdx; };

private:
  std::uint64_t m_layer{};
  std::uint8_t m_status{0};
  std::uint8_t m_detectorRegion{0};
  float m_rho{};
  float m_delta{};
  float m_r{};
  float m_eta{};
  float m_phi{};
  std::uint64_t m_clusterIndex{};
};

class CLUECalorimeterHitCollection : public DataObject {
public:
  std::vector<CLUECalorimeterHit> vect;
};

} // namespace clue

#endif
