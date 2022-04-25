#include "CLUECalorimeterHit.h"
#include <cmath>

namespace clue{

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch)
  : CalorimeterHit(ch) {
  setR();
  setEta();
  setPhi();
}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion, const int layer)
  : CalorimeterHit(ch),
    m_layer(layer),
    m_detectorRegion(detRegion) {
  setR();
  setEta();
  setPhi();
}

CLUECalorimeterHit::CLUECalorimeterHit(const CalorimeterHit& ch, const CLUECalorimeterHit::DetectorRegion detRegion, const int layer,
                                       const CLUECalorimeterHit::Status status, const int clusterIndex, const float rho, const float delta)
  : CalorimeterHit(ch),
    m_detectorRegion(detRegion),
    m_layer(layer),
    m_status(status),
    m_clusterIndex(clusterIndex),
    m_rho(rho),
    m_delta(delta) {
  setR();
  setEta();
  setPhi();
}

const std::uint64_t& CLUECalorimeterHit::getLayer() const { return m_layer; }
bool         CLUECalorimeterHit::inBarrel() const { return (m_detectorRegion == barrel ? true : false ) ; }
bool         CLUECalorimeterHit::inEndcap() const { return (m_detectorRegion == endcap ? true : false ) ; }
bool         CLUECalorimeterHit::isFollower() const { return (m_status == follower ? true : false ) ; }
bool         CLUECalorimeterHit::isSeed() const { return (m_status == seed ? true : false ) ; }
bool         CLUECalorimeterHit::isOutlier() const { return (m_status == outlier ? true : false ) ; }
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
