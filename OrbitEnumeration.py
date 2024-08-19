import numpy as np
import json
import itertools
from datetime import datetime, timedelta, timezone

class OrbitTatc():
    
    def __init__(self, semiMajorAxis, eccentricity, inclination, longAscendingNode, argPeriapsis, trueAnomaly):
        self.semiMajorAxis = semiMajorAxis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.longAscendingNode = longAscendingNode
        self.argPeriapsis = argPeriapsis
        self.trueAnomaly = trueAnomaly

class SatTatc():
    
    def __init__(self, orbit, FOV):
        self.orbit = orbit
        self.FOV = FOV


def orbit2JSON(satList,samplePoints, start, duration, analysisType):
    """
    Function to take an Orbit object (as created in the vassar python code) and create a JSON file with all the information
    """
    satDicts = []
    for sat in satList:
        orbit = sat.orbit
        orbDict = vars(orbit)
        satDicts.append({"orbit":orbDict, "FOV":sat.FOV})
    tatcDict = {"satellites":satDicts, "samplePoints":samplePoints, "start":start, "duration":duration, "analysisType":analysisType}

    # Serializing json
    jsonOutput = json.dumps(tatcDict, indent=4)

    # Writing to sample.json
    with open("coverageAnalysisCallObject" + str(ind) + ".json", "w") as outfile:
      outfile.write(jsonOutput)

    # print(jsonOrbit)
    return jsonOutput

# def JSON2Orbit(jsonOrbit):
#     """
#     Function to take a json file describing an orbit and create an orbit class (which can be used for other functions)
#     """
#     orbDict = json.loads(jsonOrbit)
#     return orbDict

# Define parameters

# Orbit
rad = 6371 # km
semiMajorAxes = np.linspace(100+rad,35786+rad,num=2) # km
# inclinations = np.linspace(-90,90,num=1) # deg
inclinations = [0]
eccentricites = np.linspace(0,1,num=1)
longAscendingNodes = np.linspace(0,360,num=1) # deg
argPeriapses = np.linspace(0,360,num=1) # deg
trueAnomalies = np.linspace(0,360,num=1) # deg

FOVs = [100,150] # Degrees
samplePoints = [[0,0,0],[40,0,0],[-40,0,0]]
# start = datetime(2024, 1, 1, tzinfo=timezone.utc)
# start = start.strftime("%Y%m%d") # yyyymmdd
start = "20240101" # yyymmdd
duration = 7*24*60*60 # 7 days in seconds
analysisType = "Example Analysis Type"

ind = 0
fullFactEnum = itertools.product(semiMajorAxes,inclinations,eccentricites,longAscendingNodes,argPeriapses,trueAnomalies,FOVs)
for vals in fullFactEnum:
    # Call TATC
    orbit = OrbitTatc(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5])
    satList = [SatTatc(orbit,vals[6]),SatTatc(orbit,vals[6])]
    jsonOrbit = orbit2JSON(satList, samplePoints, start, duration, analysisType)
    ind += 1

