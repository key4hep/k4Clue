#ifndef K4CLUE_CLUECALORIMETERHIT_H
#define K4CLUE_CLUECALORIMETERHIT_H

#include "edm4hep/CalorimeterHit.h"
using namespace edm4hep;

namespace clue {

class CLUECalorimeterHit : public CalorimeterHit {
public:
  using CalorimeterHit::CalorimeterHit;

  /// constructors
  CLUECalorimeterHit(const CalorimeterHit& ch);// : CalorimeterHit(ch) {}

  CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const float rho, const float delta);

  /// Access the layer number
  const std::uint64_t& getLayer() const;

  /// Access the delta
  const float& getDelta() const;

  /// Access the rho
  const float& getRho() const;

private:
  std::uint64_t m_layer{};
  float m_rho{};
  float m_delta{};
};

} // namespace clue

#endif
