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

def getMissionDict(semiMajorAxis, inclination, eccentricity, longAscendingNode, argPeriapsis, trueAnomaly, instrument, FOR):
    """
    Returns a dictionary of the mission parameters
    """
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
                "targetBlackBodyTemp": None,
                "detectorWidth": None,
                "maxDetectorExposureTime": None,
                "snrThreshold": None,
                "name": instrument,
                "acronym": instrument,
                "mass": None,
                "volume": None,
                "power": None,
                "orientation": {
                    "convention": "SIDE_LOOK",
                    "sideLookAngle": None,
                    "@type": "Orientation"
                },
                "fieldOfView": {
                    "sensorGeometry": "RECTANGULAR",
                    "fullConeAngle": None,
                    "alongTrackFieldOfView": None,
                    "crossTrackFieldOfView": FOR,
                    "@type": "FieldOfView"
                },
                "dataRate": 384.0,
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

def payloadMissionFFE(instruments, altitudes, inclinations):
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

    fullFactEnum = itertools.product(semiMajorAxes,inclinations,eccentricites,longAscendingNodes,argPeriapses,trueAnomalies,instruments,FOR)
    for ind, vals in enumerate(fullFactEnum):

        SCDesignDict = getMissionDict(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5],vals[6],vals[7])

        # Save to JSON
        SCDesignJSON = json.dumps(SCDesignDict, indent=4)
        with open("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", "w") as outfile:
            outfile.write(SCDesignJSON)


        # Call loadJSONSCDesign from SpacecraftDesignSelection.py
        scMass, subsMass, components, costEstimationJSONFile, coverageRequestJSONFile = loadJSONSCDesign("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", ind)

        totalMissionCost = loadJSONCostEstimationSingle(costEstimationJSONFile) 
        harmonicMeanRevisit = tatcCovReqTransformer(coverageRequestJSONFile)

        allMissionCosts.append(totalMissionCost)
        allHMeanRevisits.append(harmonicMeanRevisit)

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
        print("Harmonic Mean Revisit Time: ",harmonicMeanRevisit)

        print("\nROBOT VOICE: SIMULATION OVER\n")

    return allMissionCosts, allHMeanRevisits


climateCentricInstruments = ["ACE_ORCA", "ACE_POL", "ACE_LID", "CLAR_ERB", "ACE_CPR", "DESD_SAR", "DESD_LID", "GACM_VIS", "GACM_SWIR", "HYSP_TIR", "POSTEPS_IRS", "CNES_KaRIN"]
climateCentricAltitudes = np.arange(400,800,100)
climateCentricInclinations = [30, 60, 90, "SSO"]
allMissionCosts, allHMeanRevisits = payloadMissionFFE(climateCentricInstruments, climateCentricAltitudes, climateCentricInclinations)

