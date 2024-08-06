import numpy as np

def SABattMass(period,fracSunlight,worstSunAngle,PavgPayload,PpeakPayload,lifetime,DOD):

    # general parameters
    Xe = 0.65 # efficiency from solar arrays to battery to equipment (SMAD page 415)
    Xd = 0.85 # efficiency from solar arrays to equipment (SMAD page 415)
    P0 = 253 # in W/m2, corresponds to GaAs, see SMAD page 412
    Id = 0.77 # SMAD page 412
    degradation = 0.0275 # Degradation of solar arrays performance in % per year, corresponds to multi-junction
    specPowerSA = 25 # in W per kg see SMAD chapter 11
    n = 0.9 # Efficiency of battery to load (SMAD page 422)
    specEnergyDensityBatt = 40 # In Whr/kg see SMAD page 420, corresponds to Ni-H2

    # Solar Array numbers
    PavgPayload = PavgPayload/0.4 # SMAD page 340 to take into account bus power
    PpeakPayload = PpeakPayload/0.4 # SMAD page 340 to take into account bus power
    Pd = 0.8*PavgPayload + 0.2*PpeakPayload
    Pe = Pd
    Td = period*fracSunlight
    Te = period - Td

    # How much power is needed from the solar array
    Psa = ((Te/Xe)*Pe + (Pd*(Td/Xd)))/Td

    # What the Solar array technology can give
    theta = np.deg2rad(worstSunAngle)
    pDensityBOL = np.abs(P0*Id*np.cos(theta))
    Ld = (1-degradation)**lifetime
    pDensityEOL = pDensityBOL*Ld

    # Surface Required
    Asa = (Psa/pDensityEOL)

    # Power at BOL
    pBOL = pDensityBOL*Asa

    # Mass of the solar array
    # 1kg per 25W at BOL (See SMAD chapter 10).
    massSA = pBOL/specPowerSA

    # Batteries
    Cr = (Pe*Te)/(3600*DOD*n)
    massBatt = Cr/specEnergyDensityBatt

    return massSA, massBatt, pBOL

def estimateDepthofDischarge(orbitType):
    """
    This function estimates the depth of discharge of an orbit
    see SMAD Page 422
    """
    if orbitType == "GEO":
        DOD = 0.8
    elif orbitType == "SSO":
        DOD = 0.6
    else:
        DOD = 0.4
    return DOD


def designEPS(payloads,mission,spacecraft,compInstance):
    """
    Returns the mass of the EPS system
    """
    # pull params
    PavgPayload = sum([payload.avgPower for payload in payloads])
    PpeakPayload = sum([payload.peakPower for payload in payloads])
    fracSunlight = mission.fractionSunlight
    worstSunAngle = 0.0 # Chosen from other example, examine more closely
    period = mission.period
    lifetime = mission.lifetime
    dryMass = spacecraft.dryMass
    DOD = mission.depthOfDischarge

    massSA,massBatt,pBOL = SABattMass(period,fracSunlight,worstSunAngle,PavgPayload,PpeakPayload,lifetime,DOD)

    # Others: regulators, converters, wiring
    # SMAD page 334, assume all the power is regulated and half is converted.
    massOther = ((0.02 + 0.0125)*pBOL) + (0.02*dryMass)

    # Add all together
    EPSMass = massSA + massBatt + massOther
    EPSComps = []
    return EPSMass, EPSComps