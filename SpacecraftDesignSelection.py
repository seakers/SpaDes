import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *
from SCDesignClasses import *


def loadJSONSCDesign(jsonPath):
    """
    Function to load data for the spacecraft design module. Calls iterativeDesign
    """

    ## Load JSON files containing payload and mission data

    jsonFile = open(jsonPath)
    jsonDict = json.load(jsonFile)

    payloadDict = jsonDict['payloads']
    missionDict = jsonDict['mission']

    # import into component object
    payloads = []
    for payload in payloadDict:
        payloadComp = Component(
            type="payload",
            mass=payload['mass'],
            dimensions=payload['dimensions'],
            avgPower=payload['mass'],
            peakPower=payload['peakPower'],
            name=payload['name'],
            tempRange=payload['tempRange'],
            resolution=payload['resolution'],
            FOV=payload['FOV'],
            specRange=payload['specRange'],
            dataRate=payload['dataRate']
            )
        payloads.append(payloadComp)
    
    # import into mission object
    mission = Mission(missionDict['orbit'])

    ## Load spreadsheets containing component data

    # ADCS Components
    reactionWheelData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Reaction Wheels')
    CMGData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'CMG')
    magnetorquerData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Magnetorquers')
    starTrackerData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Star Trackers')
    sunSensorData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Sun Sensors')
    earthHorizonSensorData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Earth Horizon Sensors')
    magnetometerData = pd.read_excel('SCDesignData/ADCSData.xlsx', 'Magnetometers')

    ADCSData = {"reaction wheel":reactionWheelData,"CMG":CMGData,"magnetorquer":magnetorquerData,"star tracker":starTrackerData,
                "sun sensor":sunSensorData,"earth horizon sensor":earthHorizonSensorData,"magnetometer":magnetometerData}

    # Ground Station information
    contacts = pd.read_excel('SCDesignData/Ground Contacts.xlsx', 'Accesses')
    downlink = pd.read_excel('SCDesignData/Ground Contacts.xlsx', 'Downlink')
    uplink = pd.read_excel('SCDesignData/Ground Contacts.xlsx', 'Uplink')

    GSData = {"contacts":contacts,"downlink":downlink,"uplink":uplink}

    # launch vehicle data
    LVData = pd.read_excel('SCDesignData/LaunchVehicleData.xlsx', 'Launch Vehicles')


    scMass, subsMass, components = iterativeDesign(payloads, mission, ADCSData, GSData, LVData)

    costEstimationJSONFile = costEstimationJSON(payloads, mission, scMass, subsMass, components)

    return scMass, subsMass, components, costEstimationJSONFile


def iterativeDesign(payloads, mission, ADCSData, GSData, LVData):
    """
    Function that iteratively calls massBudget until the spacecraft mass calculation converges
    """

    # Get the prilimary mass
    payloadsMass = 0
    for payload in payloads:
        payloadsMass += payload.mass
    prevMass = prelimMassBudget(payloadsMass)

    # Create component instance for use in subsystem design
    compInstance = Component("Instance")

    # Initailize variables for the while loop
    thresh = 0.001
    newMass = prevMass
    print("\nMass Estimates:")
    print(prevMass)

    converged = False

    # While loop to iteratively find the spacecraft charactaristics. Converges when the mass stops changing
    while not converged:
        prevMass = newMass
        spacecraft = Spacecraft(prevMass,payloads,mission)
        newMass, subsMass, components = massBudget(payloads,mission,spacecraft,compInstance,ADCSData,GSData,LVData)
        print(newMass)
        # check if mass has changed above threshold
        converged = np.abs(prevMass - newMass)/newMass < thresh

    return newMass, subsMass, components

def costEstimationJSON(payloads, mission, scMass, subsMass, components):

    # pull params
    PavgPayload = sum([payload.avgPower for payload in payloads])
    PpeakPayload = sum([payload.peakPower for payload in payloads])
    fracSunlight = mission.fractionSunlight
    worstSunAngle = 0.0 # Chosen from other example, examine more closely
    period = mission.period
    lifetime = mission.lifetime
    DOD = mission.depthOfDischarge

    pBOL = SABattMass(period,fracSunlight,worstSunAngle,PavgPayload,PpeakPayload,lifetime,DOD)[2]

    data = {
        "satelliteDryMass": scMass,
        "structureMass": subsMass["Structure Mass"],
        "propulsionMass": subsMass["Propulsion Mass"],
        "ADCSMass": subsMass["ADCS Mass"],
        "avionicsMass": subsMass["Avionics Mass"],
        "thermalMass": subsMass["Thermal Mass"],
        "EPSMass": subsMass["EPS Mass"],
        "satelliteBOLPower": pBOL,
        "satDataRatePerOrbit": sum([payload.dataRate for payload in payloads]),
        "lifetime": mission.lifetime,
        "numPlanes": mission.numPlanes,
        "numSats": mission.numSats,
        "instruments": [],
        "launchVehicle": {
            "height": components["LVChoice"].height,
            "diameter": components["LVChoice"].diameter,
            "cost": components["LVChoice"].cost
        }
    }

    for payload in payloads:
        instrument_data = {
            "trl": 9, # all components currently are in a database for sale, so I am assuming they are at TRL 9
            "mass": payload.mass,
            "avgPower": payload.avgPower,
            "dataRate": payload.dataRate
        }
        data["instruments"].append(instrument_data)

    costEstJSONFilename = 'spacecraftCostEstObject.json'
    costEstJSON = json.dumps(data, indent=4)
    with open(costEstJSONFilename, 'w') as jsonFile:
        jsonFile.write(costEstJSON)
    # with open(costEstJSONFilename, 'w') as jsonFile:
    #     json.dump(data, jsonFile, indent=4)

    return costEstJSONFilename
