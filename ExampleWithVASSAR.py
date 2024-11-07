import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *
from SCDesignClasses import *
from SpacecraftDesignSelection import loadJSONSCDesign
from CostEstimationJSON import loadJSONCostEstimationSingle
from CallCoverageAnalysis import tatcCovReqTransformer
from TestScienceCalc import testScienceCalc

def getMissionDict(semiMajorAxis, inclination, eccentricity, longAscendingNode, argPeriapsis, trueAnomaly, FOR, instrumentName, instruments):
    """
    Returns a dictionary of the mission parameters
    """
    instrumentDict = instruments[instrumentName]
    # just one sat rn
    missionDict = {
        "@type": "Satellite",
        "@id": "sat-0",
        "name": "MicroMAS-2",
        "acronym": "MicroMAS-2",
        "mass": None,
        "dryMass": None,
        "volume": None,
        "power": None,
        "commBand": [
            "X"
        ],
        "payload": [
            {
                "scanTechnique": "PUSHBROOM",
                "numberOfDetectorsRowsAlongTrack": None,
                "numberOfDetectorsColsCrossTrack": None,
                "Fnum": None,
                "focalLength": None,
                "apertureDia": None,
                "operatingWavelength": None,
                "bandwidth": None,
                "opticsSysEff": None,
                "quantumEff": None,
                "numOfReadOutE": None,
                "targetBlackBodyTemp": float(np.mean(instrumentDict["tempRange"])),
                "temperatureRange": instrumentDict["tempRange"],
                "detectorWidth": None,
                "maxDetectorExposureTime": None,
                "snrThreshold": None,
                "name": instrumentName,
                "acronym": instrumentName,
                "mass": instrumentDict["mass"],
                "dimensions": instrumentDict["dimensions"],
                "volume": float(np.prod(instrumentDict["dimensions"])),
                "power": instrumentDict["avgPower"],
                "peakPower": instrumentDict["peakPower"],
                "resolution": instrumentDict["resolution"],
                "orientation": {
                    "convention": "SIDE_LOOK",
                    "sideLookAngle": None,
                    "@type": "Orientation"
                },
                "fieldOfView": {
                    "sensorGeometry": "RECTANGULAR",
                    "fullConeAngle": None,
                    "alongTrackFieldOfView": None,
                    "crossTrackFieldOfView": instrumentDict["FOV"],
                    "fieldOfRegard": FOR,
                    "@type": "FieldOfView"
                },
                "dataRate": instrumentDict["dataRate"],
                "bitsPerPixel": None,
                "techReadinessLevel": None,
                "mountType": "BODY",
                "@type": "Passive Optical Scanner"
            }
        ],
        "orbit": {
            "@type": "Orbit",
            "orbitType": "KEPLERIAN",
            "semimajorAxis": semiMajorAxis,
            "inclination": inclination,
            "eccentricity": eccentricity,
            "periapsisArgument": argPeriapsis,
            "rightAscensionAscendingNode": longAscendingNode,
            "trueAnomaly": trueAnomaly,
            "epoch": "2019-08-01T00:00:00Z"
        },
        "techReadinessLevel": 9,
        "isGroundCommand": True,
        "isSpare": False,
        "propellantType": "MONO_PROP",
        "stabilizationType": "AXIS_3"
    }

    return missionDict

def payloadMissionFFE(instruments, instrumentNames, altitudes, inclinations):
    """
    Performs a full factorial enumeration of the payload and mission options
    """
    rad = 6371 # km
    semiMajorAxes = [x+rad for x in altitudes] # km
    eccentricites = [0]
    longAscendingNodes = [0] # deg
    argPeriapses = [0] # deg
    trueAnomalies = [0] # deg
    FOR = [100] # deg

    allMissionCosts = []
    allScienceScores = []

    fullFactEnum = itertools.product(semiMajorAxes,inclinations,eccentricites,longAscendingNodes,argPeriapses,trueAnomalies,FOR,instrumentNames)
    for ind, vals in enumerate(fullFactEnum):

        SCDesignDict = getMissionDict(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5],vals[6],vals[7],instruments)

        # Save to JSON
        SCDesignJSON = json.dumps(SCDesignDict, indent=4)
        with open("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", "w") as outfile:
            outfile.write(SCDesignJSON)


        # Call loadJSONSCDesign from SpacecraftDesignSelection.py
        scMass, subsMass, components, costEstimationJSONFile, coverageRequestJSONFile = loadJSONSCDesign("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", ind)

        totalMissionCost = loadJSONCostEstimationSingle(costEstimationJSONFile) 
        harmonicMeanRevisit = tatcCovReqTransformer(coverageRequestJSONFile)
        scienceScore = testScienceCalc("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", harmonicMeanRevisit)

        allMissionCosts.append(totalMissionCost)
        allScienceScores.append(scienceScore)

        print("\nFinal Mass: ",scMass)
        print("Propulsion Mass: ",subsMass["Propulsion Mass"]," (",subsMass["Propulsion Mass"]/scMass*100,"%)")
        print("Structure Mass: ",subsMass["Structure Mass"]," (",subsMass["Structure Mass"]/scMass*100,"%)")
        print("EPS Mass: ",subsMass["EPS Mass"]," (",subsMass["EPS Mass"]/scMass*100,"%)")
        print("ADCS Mass: ",subsMass["ADCS Mass"]," (",subsMass["ADCS Mass"]/scMass*100,"%)")
        print("Avionics Mass: ",subsMass["Avionics Mass"]," (",subsMass["Avionics Mass"]/scMass*100,"%)")
        print("Payload Mass: ",subsMass["Payload Mass"]," (",subsMass["Payload Mass"]/scMass*100,"%)")
        print("Comms Mass: ",subsMass["Comms Mass"]," (",subsMass["Comms Mass"]/scMass*100,"%)")
        print("Thermal Mass: ",subsMass["Thermal Mass"]," (",subsMass["Thermal Mass"]/scMass*100,"%)\n")

        print("Total Lifecycle Cost: ", totalMissionCost)
        print("Science Score: ", scienceScore)

        print("\nROBOT VOICE: SIMULATION OVER\n")

    return allMissionCosts, allScienceScores

# made up default values for tempRange and resolution where vassar doesn't have them
climateCentricInstruments = {
    "ACE_ORCA": {
        "type": "payload",
        "mass": 137,
        "dimensions": [1, 1, 1],
        "avgPower": 132,
        "peakPower": 132,
        "name": "ACE_ORCA",
        "tempRange": [-10, 50],
        "resolution": 0.1,
        "FOV": 55,
        "specRange": "opt-VNIR",
        "dataRate": 12.0,
    },
    "ACE_POL": {
        "type": "payload",
        "mass": 132,
        "dimensions": [1, 1, 1],
        "avgPower": 152,
        "peakPower": 152,
        "name": "ACE_POL",
        "tempRange": [-10, 50],
        "resolution": 0.01,
        "FOV": 55,
        "specRange": "opt-VNIR",
        "dataRate": 15.5,
    },
    "ACE_LID": {
        "type": "payload",
        "mass": 515,
        "dimensions": [1, 1, 1],
        "avgPower": 658,
        "peakPower": 658,
        "name": "ACE_LID",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 10,
        "specRange": "opt-VNIR",
        "dataRate": 11.1,
    },
    "CLAR_ERB": {
        "type": "payload",
        "mass": 238,
        "dimensions": [2.0, 0.7, 0.8],
        "avgPower": 387,
        "peakPower": 387,
        "name": "CLAR_ERB",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 55,
        "specRange": "opt-SWIR+LWIR",
        "dataRate": 160,
    },
    "ACE_CPR": {
        "type": "payload",
        "mass": 480,
        "dimensions": [5, 5, 1],
        "avgPower": 700,
        "peakPower": 700,
        "name": "ACE_CPR",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 55,
        "specRange": "MW-94GHz MW-35GHz",
        "dataRate": 20.0,
    },
    "DESD_SAR": {
        "type": "payload",
        "mass": 400,
        "dimensions": [6.0, 6.0, 2.7],
        "avgPower": 500,
        "peakPower": 500,
        "name": "DESD_SAR",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 10,
        "specRange": "MW-L",
        "dataRate": 60,
    },
    "DESD_LID": {
        "type": "payload",
        "mass": 200,
        "dimensions": [1.5, 1.5, 2],
        "avgPower": 500,
        "peakPower": 500,
        "name": "DESD_LID",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 10,
        "specRange": "opt-NIR",
        "dataRate": 1,
    },
    "GACM_VIS": {
        "type": "payload",
        "mass": 46,
        "dimensions": [0.5, 0.4, 0.35],
        "avgPower": 56,
        "peakPower": 56,
        "name": "GACM_VIS",
        "tempRange": [-10, 50],
        "resolution": 0.2,
        "FOV": 55,
        "specRange": "opt-UV",
        "dataRate": 0.77,
    },
    "GACM_SWIR": {
        "type": "payload",
        "mass": 250,
        "dimensions": [1, 1, 1],
        "avgPower": 208,
        "peakPower": 208,
        "name": "GACM_SWIR",
        "tempRange": [-10, 50],
        "resolution": 0.2,
        "FOV": 55,
        "specRange": "opt-SWIR",
        "dataRate": 40,
    },
    "HYSP_TIR": {
        "type": "payload",
        "mass": 99,
        "dimensions": [1.2, 0.5, 0.4],
        "avgPower": 78,
        "peakPower": 78,
        "name": "HYSP_TIR",
        "tempRange": [-10, 50],
        "resolution": 0.0055,
        "FOV": 10,
        "specRange": "opt-TIR",
        "dataRate": 65,
    },
    "POSTEPS_IRS": {
        "type": "payload",
        "mass": 316,
        "dimensions": [1, 1, 1],
        "avgPower": 457,
        "peakPower": 457,
        "name": "POSTEPS_IRS",
        "tempRange": [-10, 50],
        "resolution": 0.1,
        "FOV": 55,
        "specRange": "opt-SWIR",
        "dataRate": 7.5,
    },
    "CNES_KaRIN": {
        "type": "payload",
        "mass": 300,
        "dimensions": [10, 2.2, 0.18],
        "avgPower": 300,
        "peakPower": 300,
        "name": "CNES_KaRIN",
        "tempRange": [-10, 50],
        "resolution": 10,
        "FOV": 10,
        "specRange": "MW-Ku",
        "dataRate": 10,
    }
}
climateCentricInstrumentNames = list(climateCentricInstruments.keys())
# climateCentricAltitudes = [400, 500, 600, 700, 800]
climateCentricAltitudes = [400]
# climateCentricInclinations = [30, 60, 90, "SSO"]
climateCentricInclinations = [30]
allMissionCosts, allScienceScores = payloadMissionFFE(climateCentricInstruments, climateCentricInstrumentNames, climateCentricAltitudes, climateCentricInclinations)

plt.scatter(allScienceScores,allMissionCosts)
plt.ylabel("Total Lifecycle Cost")
plt.xlabel("Science Score")
plt.title("Total Lifecycle Cost vs Science Score")
plt.show()