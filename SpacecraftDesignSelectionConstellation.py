import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *
from SCDesignClasses import *


def loadJSONConst(jsonPath): # work in progress -- need to make some decisions on how to handle multiple constellations instead
    """
    Function to load data for the spacecraft design module. Calls iterativeDesign
    Intended to call from the json files output from Pau's TSE code
    """

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


    ## Load JSON files containing payload and mission data

    jsonFile = open(jsonPath)
    jsonDict = json.load(jsonFile)
    
    scMasses = []
    subsMasses = []
    constComponents = []
    constPayloads = []
    constMissions = []

    constellationList = jsonDict["spaceSegment"]
    for constellation in constellationList:

        satelliteList = constellation["satellites"]

        if "numberPlanes" in constellation:
            numPlanes = constellation["numberPlanes"]
        else:
            numPlanes = 1
        numSats = constellation["numberSatellites"]

        # run spacecraft design on the first satellite in the constellation for now
        # this is based on the assumption that all orbits in a constellation are similar enough
        # potentially in the future combine the different missions into one.
        orbitDict = satelliteList[0]['orbit']

        # import into mission object
        mission = Mission(
            semiMajorAxis=orbitDict['semimajorAxis'],
            inclination=orbitDict['inclination'],
            eccentricity=orbitDict['eccentricity'],
            longAscendingNode=orbitDict['rightAscensionAscendingNode'],
            argPeriapsis=orbitDict['periapsisArgument'],
            trueAnomaly=orbitDict['trueAnomaly'],
            numPlanes=numPlanes,
            numSats=numSats
        )

        satPayloads = []

        for satellite in satelliteList:
            payloadList = satellite['payload']

            # import into component object
            payloads = []
            for payload in payloadList:
                payloadComp = Component( # some of these are not a 1 to 1 match with the json file, need to check
                    type="payload",
                    mass=payload['mass'],
                    dimensions=[np.power(payload['volume'],1/3),np.power(payload['volume'],1/3),np.power(payload['volume'],1/3)],
                    avgPower=payload['power'],
                    peakPower=payload['power'],
                    name=payload['name'],
                    tempRange=[payload['targetBlackBodyTemp']-25,payload['targetBlackBodyTemp']+25],
                    resolution=(payload['fieldOfView']['fullConeAngle']/payload['numberOfDetectorsColsCrossTrack'])*3600, # arcseconds
                    FOV=payload['fieldOfView']['fullConeAngle'],
                    specRange=payload['operatingWavelength'],
                    dataRate=payload['dataRate'],
                    FOR=payload['fieldOfView']['crossTrackFieldOfView']
                    # swathWidth=payload['swathWidth']
                    )
                payloads.append(payloadComp)
            satPayloads.append(payloads)

        # run spacecraft design on the first satellite in the constellation for now
        # potentially in the future combine the different missions into one.
        scMass, subsMass, components = iterativeDesign(satPayloads[0], mission, ADCSData, GSData, LVData)
        scMasses.append(scMass)
        subsMasses.append(subsMass)
        constComponents.append(components)
        constMissions.append(mission)
        constPayloads.append(satPayloads)
    

    costEstimationJSONFile = costEstimationJSON(constPayloads, constMissions, scMasses, subsMasses, constComponents)

        # coverageRequestJSONFile = coverageRequestJSON(constPayloads[0], mission, ind)

        # ind+=1

    # return scMasses, subsMasses, constComponents, costEstimationJSONFile, coverageRequestJSONFile
    return scMasses, subsMasses, constComponents, costEstimationJSONFile


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

def costEstimationJSON(constPayloads, constMissions, scMasses, subsMasses, constComponents):

    costCallDict = {'constellations':[]}
    ind = 0

    for ind, (satPayloads, mission, scMass, subsMass, components) in enumerate(zip(constPayloads, constMissions, scMasses, subsMasses, constComponents)):

        payloads = satPayloads[0] # for now just use the first satellite in the constellation

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
            "constellationInd": ind,
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

        ind+=1

        for payload in payloads:
            instrument_data = {
                "trl": 9, # all components currently are in a database for sale, so I am assuming they are at TRL 9
                "mass": payload.mass,
                "avgPower": payload.avgPower,
                "dataRate": payload.dataRate
            }
            data["instruments"].append(instrument_data)

        costCallDict['constellations'].append(data)

    costEstJSONFilename = 'JsonFiles\constellationCostEstObject.json'
    costEstJSON = json.dumps(costCallDict, indent=4)
    with open(costEstJSONFilename, 'w') as jsonFile:
        jsonFile.write(costEstJSON)
    # with open(costEstJSONFilename, 'w') as jsonFile:
    #     json.dump(data, jsonFile, indent=4)

    return costEstJSONFilename

def coverageRequestJSON(payloads, mission, ind):

    FOR = max([payload.FOR + payload.FOV for payload in payloads])

    missionDict = {
        "semiMajorAxis": mission.a,
        "inclination": mission.i,
        "eccentricity": mission.e,
        "longAscendingNode": mission.lan,
        "argPeriapsis": mission.argp,
        "trueAnomaly": mission.trueAnomaly
    }

    satDicts = [{
        "orbit": missionDict,
        "FOR": FOR
        # "swathWidth": swathWidth
    }]

    samplePoints = {
        "type": "grid", # can be grid or points
        "deltaLatitude": 5,
        "deltaLongitude": 5,
        "region": [-180,-50,180,50] # [lon1,lat1,lon2,lat2]
    }

    # samplePoints = {
    #     "type": "points", # can be grid or points
    #     "points": [[0,0],[0,20],[0,-20],[20,0],[-20,0],[20,20],[20,-20],[-20,20],[-20,-20]] # [lon,lat]
    # }

    start = "20240101" # YYYYMMDD

    duration = 7*24*60*60 # one week in seconds

    analysisType = 'U'

    tatcDict = {
        "satellites":satDicts, 
        "samplePoints":samplePoints, 
        "start":start, 
        "duration":duration, 
        "analysisType":analysisType
    }

    # Serializing json
    jsonOutput = json.dumps(tatcDict, indent=4)

    coverageJSONFile = "JsonFiles\coverageAnalysisCallObject" + str(ind) + ".json"

    # Writing to sample.json
    with open(coverageJSONFile, "w") as outfile:
      outfile.write(jsonOutput)

    # print(jsonOrbit)
    return coverageJSONFile
