import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *
from SCDesignClasses import *
from SpacecraftDesignSelection import loadJSONSCDesign
from CostEstimationJSON import loadJSONCostEstimation
from CallCoverageAnalysis import tatcCovReqTransformer

### Main script for running the SpaDes tool. Takes in a JSON file with an orbit and a payload (chosen randomly currently for testing) 
### and outputs the mass of the spacecraft and the mass of the subsystems, along with a list of components. This is then used to estimate the cost of the mission.
### Interaction between the different modules is done through JSON files because this allows for more flexibility, though just passing objects around would probably be faster.

### Goes through a full factorial enumeration of the payload and mission options to find the best combination

def choosePayloadJSON(payloadData, payloadInd):
    """
    Chooses a random payload, extracts the data from the PayloadData Excel sheet, and makes a component object with that data
    """
    # payloadInd = np.random.randint(payloadData.shape[0])
    
    # Setting the payload for testing purposes
    # payloadInd = 0

    # Extracting the data into a dict then a JSON
    payloadDict = {
        'mass':float(payloadData.loc[payloadInd,'Mass (kg)']),
        'dimensions':[float(payloadData.loc[payloadInd,'Length (m)']),float(payloadData.loc[payloadInd,'Width (m)']),float(payloadData.loc[payloadInd,'Height (m)'])],
        'avgPower':float(payloadData.loc[payloadInd,'Avg Power (W)']),
        'peakPower':float(payloadData.loc[payloadInd,'Peak Power (W)']),
        'name':payloadData.loc[payloadInd,'Name'],
        'tempRange':[float(payloadData.loc[payloadInd,'Temp Min (C)']),float(payloadData.loc[payloadInd,'Temp Max (C)'])],
        'resolution':float(payloadData.loc[payloadInd,'Resolution (arcsec)']),
        'FOV':float(payloadData.loc[payloadInd,'FOV (degrees)']),
        'specRange':payloadData.loc[payloadInd,'Spectral Range'],
        'dataRate':float(payloadData.loc[payloadInd,'Data Rate (Mbps)'])
    } # need to change all numbers to float because they are read in as numpy types which JSON cannot handle

    print("Payload Chosen: ",payloadDict['name'])

    return payloadDict

def chooseMissionJSON(semiMajorAxis, inclination, eccentricity, longAscendingNode, argPeriapsis, trueAnomaly):
    """
    Function to randomly choose an orbit and create a mission object
    """
    # Currently choses uniformly. Later should weight it based on the payload chosen
    # orbitOptions = ["LEO", "SSO", "MEO", "GEO"]
    # orbit = orbitOptions[np.random.randint(len(orbitOptions))]

    # Setting the orbit for testing purposes
    # orbit = "GEO"

    missionDict = {
        "semiMajorAxis":semiMajorAxis,
        "inclination":inclination,
        "eccentricity":eccentricity,
        "longAscendingNode":longAscendingNode,
        "argPeriapsis":argPeriapsis,
        "trueAnomaly":trueAnomaly
    }

    print("Mission Chosen: ",missionDict)

    return missionDict


def payloadMissionFFE(payloadData):
    """
    Performs a full factorial enumeration of the payload and mission options
    """

    # Define parameters

    payloads = np.arange(len(payloadData['camera'])) # Index of the payload to choose

    # Orbit
    rad = 6371 # km
    semiMajorAxes = [200+rad,400+rad] # km
    inclinations = [60]
    eccentricites = [0]
    longAscendingNodes = [0] # deg
    argPeriapses = [0] # deg
    trueAnomalies = [0] # deg

    allMissionCosts = []
    allHMeanRevisits = []

    fullFactEnum = itertools.product(semiMajorAxes,inclinations,eccentricites,longAscendingNodes,argPeriapses,trueAnomalies,payloads)
    for ind, vals in enumerate(fullFactEnum):
        missionDict = chooseMissionJSON(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5])
        payloadDict = choosePayloadJSON(payloadData['camera'], vals[6])

        SCDesignDict = {
            'payloads':[payloadDict],
            'mission':missionDict
        }

        # Save to JSON
        SCDesingJSON = json.dumps(SCDesignDict, indent=4)
        with open("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", "w") as outfile:
            outfile.write(SCDesingJSON)


        # Call loadJSONSCDesign from SpacecraftDesignSelection.py
        scMass, subsMass, components, costEstimationJSONFile, coverageRequestJSONFile = loadJSONSCDesign("JsonFiles\spacecraftDesignCallObject" + str(ind) + ".json", ind)

        totalMissionCost = loadJSONCostEstimation(costEstimationJSONFile) 
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



### Initialization

rand = np.random.default_rng()

## Load all the component catalogs
# Payloads
cameraData = pd.read_excel('SCDesignData/PayloadData.xlsx', 'Cameras')

payloadData = {"camera":cameraData}

allMissionCosts, allHMeanRevisits = payloadMissionFFE(payloadData)

plt.scatter(allMissionCosts,allHMeanRevisits)
plt.xlabel("Total Lifecycle Cost")
plt.ylabel("Harmonic Mean Revisit Time (hr)")
plt.title("Total Lifecycle Cost vs Harmonic Mean Revisit Time")
plt.show()




