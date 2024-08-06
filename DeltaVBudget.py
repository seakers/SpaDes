import numpy as np
from OrbitCalculations import orbitVelocity

def computeDV(rp1,ra1,rp2,ra2,r,mu):
    """
    Computes the deltaV for an arbitrary burn
    """
    a1 = (rp1 + ra1)/2
    a2 = (rp2 + ra2)/2
    delV = np.abs(orbitVelocity(r,a2,mu) - orbitVelocity(r,a1,mu))
    return delV

def computeDelVInj(orbitType,a,rad,mu):
    """
    This rule computes the delta-V required for injection for GEO or MEO assuming a transfer orbit with a perigee 
    of 150km and an apogee at the desired orbit, as suggested in De Weck's paper found in 
    http://strategic.mit.edu/docs/2_3_JSR_parametric_NGSO.pdf. For LEO/SSO, no injection is required.
    """
    parkOrbAlt = 150000
    if orbitType == "LEO" or orbitType == "SSO":
        delV = 0
    elif orbitType == "MEO" or orbitType == "GEO":
        delV = computeDV(rad+parkOrbAlt,a,a,a,a,mu)
    return delV

def computeDelVDrag(a,e,rad):
    """
    This rule computes the delta-V required to overcome drag. The data comes from 
    De Weck's paper found in http://strategic.mit.edu/docs/2_3_JSR_parametric_NGSO.pdf
    """
    hp = (a*(1-e) - rad)/1000
    if hp < 500:
        delV = 12
    elif hp < 600:
        delV = 5
    elif hp < 1000:
        delV = 2
    elif hp >= 1000:
        delV = 0
    return delV

def computeDelVADCS(ADCSType):
    """
    This rule computes the delta-V required for attitude control. The data comes from 
    De Weck's paper found in http://strategic.mit.edu/docs/2_3_JSR_parametric_NGSO.pdf
    """
    if ADCSType == "threeAxis":
        delV = 20
    elif ADCSType == "gravGradient" or ADCSType == "spinner":
        delV = 0
    return delV

def computeDelVDeorbit(orbitType,a,dims,m,rad,mu):
    """
    This rule chooses a deorbit method and calculate the dv required
    """
    A = dims[0]*dims[1]
    if orbitType == "LEO" or orbitType == "SSO":
        deorbitType = "dragBased"
        delV = computeDV(a,a,rad,a,a,mu)
    elif orbitType == "MEO" or orbitType == "GEO":
        deorbitType = "graveyard"
        Cr = 1.5
        dh = (A/m)*Cr*1e6 + 235000
        delV = computeDV(a,a,a,a+dh,a,mu)
    return delV

def computeDeltaV(orbitType,ADCSType,a,e,dims,m,rad,mu):
    """
    Computes the total delta V by summing all other deltaVs
    """
    injDelV = computeDelVInj(orbitType,a,rad,mu)
    ADCSDelV = computeDelVADCS(ADCSType)
    dragDelV = computeDelVDrag(a,e,rad)
    deorbitDelV = computeDelVDeorbit(orbitType,a,dims,m,rad,mu)
    totalDelV = injDelV + ADCSDelV + dragDelV + deorbitDelV
    return totalDelV, injDelV
    