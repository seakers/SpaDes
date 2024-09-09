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

    allCosts = []

    constellations = data['constellations']
    for const in constellations:

        mission = Mission(
            satelliteDryMass=const['satelliteDryMass'],
            structureMass=const['structureMass'],
            propulsionMass=const['propulsionMass'],
            ADCSMass=const['ADCSMass'],
            avionicsMass=const['avionicsMass'],
            thermalMass=const['thermalMass'],
            EPSMass=const['EPSMass'],
            satelliteBOLPower=const['satelliteBOLPower'],
            satDataRatePerOrbit=const['satDataRatePerOrbit'],
            lifetime=const['lifetime'],
            numPlanes=const['numPlanes'],
            numSats=const['numSats'],
            instruments=[
                Instrument(
                    trl=instrument['trl'],
                    mass=instrument['mass'],
                    avgPower=instrument['avgPower'],
                    dataRate=instrument['dataRate']
                ) for instrument in const['instruments']
            ],
            launchVehicle=LaunchVehicle(
                height=const['launchVehicle']['height'],
                diameter=const['launchVehicle']['diameter'],
                cost=const['launchVehicle']['cost']
            )
        )

        mission = costEstimationManager(mission)

        missionCost = mission.lifecycleCost
        allCosts.append(missionCost)

    return allCosts

# Create a JSON file to input into the function
# data = {
#     "satelliteDryMass": 1000,
#     "structureMass": 500,
#     "propulsionMass": 200,
#     "ADCSMass": 100,
#     "avionicsMass": 200,
#     "thermalMass": 150,
#     "EPSMass": 100,
#     "satelliteBOLPower": 500,
#     "satDataRatePerOrbit": 100,
#     "lifetime": 10,
#     "numPlanes": 1,
#     "numSats": 1,
#     "instruments": [
#         {
#             "trl": 9,
#             "mass": 50,
#             "avgPower": 100,
#             "dataRate": 50
#         },
#         {
#             "trl": 7,
#             "mass": 30,
#             "avgPower": 80,
#             "dataRate": 30
#         }
#     ],
#     "launchVehicle": {
#         "height": 70,
#         "diameter": 3.7,
#         "cost": 62000000
#     }
# }

# with open('spacecraftCostEstObject.json', 'w') as file:
#     json.dump(data, file)

# mission = loadJSONCostEstimation('spacecraftCostEstObject.json')

