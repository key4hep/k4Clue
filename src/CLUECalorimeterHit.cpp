#include "CLUECalorimeterHit.h"
#include <cmath>

namespace clue{

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch)
  : CalorimeterHit(ch) {
  setR();
  setEta();
  setPhi();
}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const int layer, const bool inBarrel,
                                       const bool isSeed, const float rho, const float delta)
  : CalorimeterHit(ch),
    m_layer(layer),
    m_inBarrel(inBarrel),
    m_isSeed(isSeed),
    m_rho(rho),
    m_delta(delta) {
  setR();
  setEta();
  setPhi();
}

const std::uint64_t& CLUECalorimeterHit::getLayer() const { return m_layer; }
const bool&  CLUECalorimeterHit::inBarrel() const { return m_inBarrel; }
const bool&  CLUECalorimeterHit::isSeed() const { return m_isSeed; }
const float& CLUECalorimeterHit::getRho() const { return m_rho; }
const float& CLUECalorimeterHit::getDelta() const { return m_delta; }
const float& CLUECalorimeterHit::getR() const { return m_r; }
const float& CLUECalorimeterHit::getEta() const { return m_eta; }
const float& CLUECalorimeterHit::getPhi() const { return m_phi; }

void CLUECalorimeterHit::setEta() { 
  m_eta = - 1. * log(tan(atan2(m_r, getPosition().z)/2.));
}

void CLUECalorimeterHit::setPhi() {
  m_phi = atan2(getPosition().y, getPosition().x);
}

void CLUECalorimeterHit::setR() { 
  m_r = float(sqrt(getPosition().x*getPosition().x + getPosition().y*getPosition().y));
}

}
