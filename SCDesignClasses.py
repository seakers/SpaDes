from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *

### Define Classes used to store information and for dependant calcuations

class Mission:
    """
    A mission object containing the mission requirements

    Input variables and thier desired units/type:
    orbitType string
    """
    def __init__(self, semiMajorAxis, inclination, eccentricity, longAscendingNode, argPeriapsis, trueAnomaly, numPlanes, numSats):
        self.a = semiMajorAxis
        self.e = eccentricity
        self.i = inclination
        self.lan = longAscendingNode
        self.argp = argPeriapsis
        self.trueAnomaly = trueAnomaly
        self.numPlanes = numPlanes
        self.numSats = numSats
        # assuming earth orbit for now
        self.mu = 3.986e14 # m^3 s^-2
        self.rad = 6371 # km
        self.getOrbitParams()

    def getOrbitParams(self):
        if self.a < 2000 + self.rad:
            self.orbitType = "LEO"
        elif self.a < 35786 + self.rad:
            self.orbitType = "MEO"
        else:
            self.orbitType = "GEO"
        self.period = orbitPeriod(self.a,self.mu)
        self.h = rToh(self.a,self.rad) # km
        self.fractionSunlight = estimateFractionSunlight(self.a,self.rad)
        self.depthOfDischarge = estimateDepthofDischarge(self.orbitType)
        self.lifetime = 5 # chosen at random for now
        
class Spacecraft:
    """
    A spacecraft object containing aggregate charactaristics of the spacecraft
    as well as all the component objects

    Input variables and thier desired units/type:
    dryMass kg
    paylaod Component Object
    mission Mission Object
    """
    def __init__(self, dryMass, payloads, mission):
        self.dryMass = dryMass
        self.payloads = payloads
        self.mission = mission

        # Calculating preliminary spacecraft charactaristics needed for the iterative design
        self.getInitDims()
        self.getPropellantMass()
        self.getMOI()
        self.getDragCoef()
        self.getResDipole()

    def getInitDims(self):
        self.dimensions = estimateSatelliteDimensions(self.dryMass)

    def getADCS(self):
        self.ADCSType = setADCSType(self.payloads, self.mission.orbitType)

    def getDeltaV(self):
        self.deltaV, self.deltaVInj = computeDeltaV(self.mission.orbitType,self.ADCSType,self.mission.a,self.mission.e,self.dimensions,self.dryMass,self.mission.rad,self.mission.mu)
        self.deltaVADCS = self.deltaV - self.deltaVInj

    def getIsp(self):
        self.IspInj = findPropIsp(self.deltaVInj)
        self.IspADCS = findPropIsp(self.deltaVADCS)

    def getPropellantMass(self):
        self.getADCS()
        self.getDeltaV()
        self.getIsp()
        self.propellantMassInj = propMassDry(self.deltaVInj,self.IspInj,self.dryMass)
        self.propellantMassADCS = propMassDry(self.deltaVADCS,self.IspADCS,self.dryMass)
        self.wetMass = self.dryMass + self.propellantMassADCS + self.propellantMassInj

    def getMOI(self):
        self.MOI = cuboidMOI(self.wetMass,self.dimensions)

    def getDragCoef(self):
        self.Cd = estimateDragCoef(self.dimensions)
        
    def getResDipole(self):
        self.resDipole = estimateResidualDipole()

class Component:
    """
    A component object containing the component charactaristics

    Input variables and thier desired units/type:

    Necessary Variables:
    mass kg
    dimensions [m,m,m]

    Optional General Variables:
    avgPower W
    peakPower W
    name string
    tempRange [C,C]

    Used for Camera Payloads:
    resolution arcsec
    FOV degrees
    specRange string

    Used for Attitude Determination Components:
    accuracy degrees

    Used for Attitude Controllers:
    momentum Nms
    """     
    def __init__(self, type, mass=None, dimensions=None, **kwargs):
        self.type = type
        self.mass = mass # kg
        self.dimensions = dimensions # m

        # get the optional arguments which can differ between components
        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])