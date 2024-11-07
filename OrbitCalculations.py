import numpy as np

def orbitVelocity(r,a,mu):
    """
    Returns the velocity for a location in a given orbit
    """
    vel = np.sqrt(((2/r)-(1/a))*mu)
    return vel

def orbitPeriod(a,mu):
    """
    Returns the period of a given orbit
    """
    period = 2*np.pi*np.sqrt((a**3)/mu)
    return period

def atmosphericDensity(h):
    """
    Calculates rho in kg/m^3 as a function of h in m
    """
    # rho = 1e-5 * np.exp(((h/1000) - 85)/-33.387)

    # Using a table from new SMAD
    densityTable = np.array([[0,100,150,175,200,225,250,275,300,325,350,375,400,450,500,550,600,650,700,750,700,750,900,950,1000],
                            [1.2,5.67e-07,2.21e-09,9.21e-41,3.84e-10,2.12e-10,1.17e-10,7.17e-11,4.39e-11,2.85e-11,1.85e-11,1.25e-11,8.43e-12,4.05e-12,2.03e-12,1.05e-12,5.63e-13,3.08e-13,1.73e-13,9.95e-14,5.88e-14,3.57e-14,2.25e-14,1.46e-14,9.91e-15]])
    diffAltTable = h/1000 - densityTable[0]
    altIndex = np.where(diffAltTable > 0, diffAltTable, np.inf).argmin()
    density = densityTable[1][altIndex]
    return density

def rToh(r,rad):
    """
    Converts distance from center of planet to altitude
    """
    h = r - rad
    return h

def hTor(h,rad):
    """
    Converts altitude to distance from center of planet
    """
    r = h + rad
    return r

def getSemimajorAxis(orbitType,rad):
    """
    Function for getting the semimajor axis from an orbit type
    """
    # Add randomization later over likely orbits
    if orbitType == "LEO":
        h = 200000 # m
    elif orbitType == "SSO":
        h = 894000 # m
    elif orbitType == "MEO":
        h = 20000000 # m
    elif orbitType == "GEO":
        h = 35786000 # m

    a = hTor(h,rad)
    return a

def getInclination(orbitType):
    """
    Gets the inclination from an orbit type
    """
    # Add more randomization and options
    if orbitType == "SSO":
        i = 99.0*np.pi/180 # rad
    else:
        i = 30*np.pi/180 # rad, roughly the inclination of cape canaveral
    return i

def earthSubtendAngle(a,rad):
    """
    This function returns the angle in degrees subtended by the Earth from 
    the orbit
    """
    rho = np.arcsin(rad/a)
    return rho

def estimateFractionSunlight(a,rad):
    """
    Estimate fraction of sunlight based on circular orbit
    """
    if a < 7000000:
        rho = earthSubtendAngle(a,rad)
        Bs = 25
        phi = 2*np.arccos(np.cos(rho)/np.cos(Bs))
        frac = 1 - (phi/360)
    else:
        frac = 0.99
    return frac