#ifndef K4CLUE_CLUECALORIMETERHIT_H
#define K4CLUE_CLUECALORIMETERHIT_H

#include "edm4hep/CalorimeterHit.h"
using namespace edm4hep;

namespace clue {

class CLUECalorimeterHit : public CalorimeterHit {
public:
  using CalorimeterHit::CalorimeterHit;

  /// constructors
  CLUECalorimeterHit(const CalorimeterHit& ch);

  CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const bool inBarrel, 
                     const bool isSeed, const float rho, const float delta);

  /// Access the layer number
  const std::uint64_t& getLayer() const;

  /// Access the part of calorimeter
  const bool& inBarrel() const;

  /// Access the seed value
  const bool& isSeed() const;

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

private:
  std::uint64_t m_layer{};
  bool  m_inBarrel{};
  bool  m_isSeed{};
  float m_rho{};
  float m_delta{};
  float m_r{};
  float m_eta{};
  float m_phi{};
};

} // namespace clue

#endif
