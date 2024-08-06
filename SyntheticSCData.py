import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MassBudget import *
from OrbitCalculations import *
from DeltaVBudget import *
from EPSDesign import *
from SCDesignClasses import *

### Define functions for chosing the payload and mission randomly

def choosePayload(payloadData):
    """
    Chooses a random payload, extracts the data from the PayloadData Excel sheet, and makes a component object with that data
    """
    payloadInd = np.random.randint(payloadData.shape[0])
    
    # Setting the payload for testing purposes
    # payloadInd = 0

    # Extracting data from the pandas dataframe for more readable code
    pdMass = payloadData.loc[payloadInd,'Mass (kg)']
    pdDims = [payloadData.loc[payloadInd,'Length (m)'],payloadData.loc[payloadInd,'Width (m)'],payloadData.loc[payloadInd,'Height (m)']]
    pdAvgPower = payloadData.loc[payloadInd,'Avg Power (W)']
    pdPeakPower = payloadData.loc[payloadInd,'Peak Power (W)']
    pdName = payloadData.loc[payloadInd,'Name']
    pdTempRange = [payloadData.loc[payloadInd,'Temp Min (C)'],payloadData.loc[payloadInd,'Temp Max (C)']]
    pdResolution = payloadData.loc[payloadInd,'Resolution (arcsec)']
    pdFOV = payloadData.loc[payloadInd,'FOV (degrees)']
    pdSpecRange = payloadData.loc[payloadInd,'Spectral Range']
    pdDataRate = payloadData.loc[payloadInd,'Data Rate (Mbps)']

    payload = Component("payload",pdMass,pdDims,avgPower=pdAvgPower,peakPower=pdPeakPower,name=pdName,tempRange=pdTempRange,resolution=pdResolution,FOV=pdFOV,specRange=pdSpecRange,dataRate=pdDataRate)
    return payload

def chooseMission():
    """
    Function to randomly choose an orbit and create a mission object
    """
    # Currently choses uniformly. Later should weight it based on the payload chosen
    orbitOptions = ["LEO", "SSO", "MEO", "GEO"]
    orbit = orbitOptions[np.random.randint(len(orbitOptions))]

    # Setting the orbit for testing purposes
    # orbit = "GEO"

    mission = Mission(orbit)

    return mission


### Initialization and design loop

rand = np.random.default_rng()

## Load all the component catalogs
# Payloads
cameraData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/PayloadData.xlsx', 'Cameras')

payloadData = {"camera":cameraData}

# ADCS Components
reactionWheelData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Reaction Wheels')
CMGData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'CMG')
magnetorquerData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Magnetorquers')
starTrackerData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Star Trackers')
sunSensorData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Sun Sensors')
earthHorizonSensorData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Earth Horizon Sensors')
magnetometerData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/ADCSData.xlsx', 'Magnetometers')

ADCSData = {"reaction wheel":reactionWheelData,"CMG":CMGData,"magnetorquer":magnetorquerData,"star tracker":starTrackerData,
            "sun sensor":sunSensorData,"earth horizon sensor":earthHorizonSensorData,"magnetometer":magnetometerData}

# Ground Station information
contacts = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/Ground Contacts.xlsx', 'Accesses')
downlink = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/Ground Contacts.xlsx', 'Downlink')
uplink = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/Ground Contacts.xlsx', 'Uplink')

GSData = {"contacts":contacts,"downlink":downlink,"uplink":uplink}

# Launch Vehicle Data
LVData = pd.read_excel('C:/Users/snagg/Desktop/SCDesignData/LaunchVehicleData.xlsx', 'Launch Vehicles')

# Choose payload and orbit first and base everything else on that
payload = choosePayload(payloadData["camera"])
mission = chooseMission()

print("Payload: ",payload.name)
print("Orbit: ",mission.orbitType)

# Get the prilimary mass
prevMass = prelimMassBudget(payload.mass)

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
    spacecraft = Spacecraft(prevMass,payload,mission)
    newMass, subsMass, components = massBudget(payload,mission,spacecraft,compInstance,ADCSData,GSData,LVData)
    print(newMass)
    # check if mass has changed above threshold
    converged = np.abs(prevMass - newMass)/newMass < thresh

print("\nFinal Mass: ",newMass)
print("Propulsion Mass: ",subsMass["Propulsion Mass"]," (",subsMass["Propulsion Mass"]/newMass*100,"%)")
print("Structure Mass: ",subsMass["Structure Mass"]," (",subsMass["Structure Mass"]/newMass*100,"%)")
print("EPS Mass: ",subsMass["EPS Mass"]," (",subsMass["EPS Mass"]/newMass*100,"%)")
print("ADCS Mass: ",subsMass["ADCS Mass"]," (",subsMass["ADCS Mass"]/newMass*100,"%)")
print("Avionics Mass: ",subsMass["Avionics Mass"]," (",subsMass["Avionics Mass"]/newMass*100,"%)")
print("Payload Mass: ",subsMass["Payload Mass"]," (",subsMass["Payload Mass"]/newMass*100,"%)")
print("Comms Mass: ",subsMass["Comms Mass"]," (",subsMass["Comms Mass"]/newMass*100,"%)")
print("Thermal Mass: ",subsMass["Thermal Mass"]," (",subsMass["Thermal Mass"]/newMass*100,"%)")
print("Launch Vehicle: ",components["LVChoice"].name)

print("\nROBOT VOICE: SIMULATION OVER\n")

