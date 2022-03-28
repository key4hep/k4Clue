#include "CLUECalorimeterHit.h"

namespace clue{

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch)
  : CalorimeterHit(ch) {}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const float rho, const float delta)
  : CalorimeterHit(ch),
    m_layer(layer),
    m_rho(rho),
    m_delta(delta) {}

const std::uint64_t& CLUECalorimeterHit::getLayer() const { return m_layer; }
const float& CLUECalorimeterHit::getRho() const { return m_rho; }
const float& CLUECalorimeterHit::getDelta() const { return m_delta; }

}
