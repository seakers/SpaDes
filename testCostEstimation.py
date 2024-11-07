import numpy as np
import pandas as pd
from copy import deepcopy
from CostEstimation import *

missionTest = Mission(
    satelliteDryMass=5000,
    structureMass=1000,
    propulsionMass=500,
    ADCSMass=200,
    avionicsMass=300,
    thermalMass=200,
    EPSMass=400,
    satelliteBOLPower=5000,
    satDataRatePerOrbit=1000,
    lifetime=10,
    numPlanes=3,
    numSats=2,
    instruments=[
        Instrument(
            trl=5,
            mass=100,
            avgPower=100,
            dataRate=1000
        ),
        Instrument(
            trl=3,
            mass=200,
            avgPower=200,
            dataRate=1000
        )
    ],
    launchVehicle=LaunchVehicle(
        height=3.85, 
        diameter=5.06, 
        cost=45
    )
)


# get launch cost (need to put in later)

# Final Goal lifecycleCost
estimate_bus_TFU_recurring_cost(missionTest)
estimate_bus_non_recurring_cost(missionTest)
estimate_payload_cost(missionTest)
estimate_spacecraft_cost_dedicated(missionTest)
estimate_program_overhead_cost(missionTest)
estimate_integration_and_testing_cost(missionTest)
estimate_operations_cost(missionTest)

# total recurring and nonrecurring cost for total lifecycle cost
estimate_total_mission_cost_recurring(missionTest)
estimate_total_mission_cost_non_recurring(missionTest)

# total lifecycle cost
estimate_lifecycle_mission_cost(missionTest)


print("Total Lifecycle Cost: ", missionTest.lifecycleCost)
print("Total Recurring Cost: ", missionTest.missionRecurringCost)
print("Total Non-Recurring Cost: ", missionTest.missionNonRecurringCost)