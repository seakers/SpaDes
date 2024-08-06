import numpy as np
import pandas as pd
import json
from CostEstimation import *

def loadJSONCostEstimation(jsonPath):
    """
    Function to load data for the spacecraft cost estimation module
    """

    with open(jsonPath) as file:
        data = json.load(file)

    mission = Mission(
        satelliteDryMass=data['satelliteDryMass'],
        structureMass=data['structureMass'],
        propulsionMass=data['propulsionMass'],
        ADCSMass=data['ADCSMass'],
        avionicsMass=data['avionicsMass'],
        thermalMass=data['thermalMass'],
        EPSMass=data['EPSMass'],
        satelliteBOLPower=data['satelliteBOLPower'],
        satDataRatePerOrbit=data['satDataRatePerOrbit'],
        lifetime=data['lifetime'],
        numPlanes=data['numPlanes'],
        numSats=data['numSats'],
        instruments=[
            Instrument(
                trl=instrument['trl'],
                mass=instrument['mass'],
                avgPower=instrument['avgPower'],
                dataRate=instrument['dataRate']
            ) for instrument in data['instruments']
        ],
        launchVehicle=LaunchVehicle(
            height=data['launchVehicle']['height'],
            diameter=data['launchVehicle']['diameter'],
            cost=data['launchVehicle']['cost']
        )
    )

    mission = costEstimationManager(mission)

    return mission

# Create a JSON file to input into the function
data = {
    "satelliteDryMass": 1000,
    "structureMass": 500,
    "propulsionMass": 200,
    "ADCSMass": 100,
    "avionicsMass": 200,
    "thermalMass": 150,
    "EPSMass": 100,
    "satelliteBOLPower": 500,
    "satDataRatePerOrbit": 100,
    "lifetime": 10,
    "numPlanes": 1,
    "numSats": 1,
    "instruments": [
        {
            "trl": 9,
            "mass": 50,
            "avgPower": 100,
            "dataRate": 50
        },
        {
            "trl": 7,
            "mass": 30,
            "avgPower": 80,
            "dataRate": 30
        }
    ],
    "launchVehicle": {
        "height": 70,
        "diameter": 3.7,
        "cost": 62000000
    }
}

with open('Research\SpaDes\spacecraftCostEstObject.json', 'w') as file:
    json.dump(data, file)

mission = loadJSONCostEstimation('Research\SpaDes\spacecraftCostEstObject.json')

