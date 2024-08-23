import numpy as np
import json
import itertools
from datetime import datetime, timedelta, timezone
from CallCoverageAnalysis import tatcCovReqTransformer

class OrbitTatc():
    
    def __init__(self, semiMajorAxis, inclination, eccentricity, longAscendingNode, argPeriapsis, trueAnomaly):
        self.semiMajorAxis = semiMajorAxis
        self.inclination = inclination
        self.eccentricity = eccentricity
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
    with open("JsonFiles\coverageAnalysisCallObject" + str(ind) + "Test" + ".json", "w") as outfile:
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
semiMajorAxes = [200+rad,300+rad] # km
inclinations = [50,60]
eccentricites = [0,.1]
longAscendingNodes = [0,90] # deg
argPeriapses = [0,90] # deg
trueAnomalies = [0,90] # deg

FOVs = [100,150] # Degrees
# samplePoints = {
#         "type": "points", # can be grid or points
#         "points": [[0,0],[0,20],[0,-20],[20,0],[-20,0],[20,20],[20,-20],[-20,20],[-20,-20]] # [lon,lat]
#     }
samplePoints = {
        "type": "grid", # can be grid or points
        "deltaLatitude": 20,
        "deltaLongitude": 20,
        "region": [-180,-50,180,50] # [lon1,lat1,lon2,lat2]
    }
# start = datetime(2024, 1, 1, tzinfo=timezone.utc)
# start = start.strftime("%Y%m%d") # yyyymmdd
start = "20240101" # yyymmdd
duration = 7*24*60*60 # 7 days in seconds
analysisType = "U"

ind = 0
fullFactEnum = itertools.product(semiMajorAxes,inclinations,eccentricites,longAscendingNodes,argPeriapses,trueAnomalies,FOVs)
for vals in fullFactEnum:
    # Call TATC
    orbit = OrbitTatc(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5])
    satList = [SatTatc(orbit,vals[6])]
    jsonOrbit = orbit2JSON(satList, samplePoints, start, duration, analysisType)

    harmonicMeanRevisit = tatcCovReqTransformer("JsonFiles\coverageAnalysisCallObject" + str(ind) + "Test" + ".json")
    print("Harmonic Mean Revisit: ",harmonicMeanRevisit)
    ind += 1
