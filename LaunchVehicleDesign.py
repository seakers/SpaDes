import numpy as np
from OrbitCalculations import rToh

# def large_enough_height(lv, dim):
#     fairing_dimensions = lv['height']
#     diam = dim[0]
#     if diam is None:
#         return 0
#     if max(fairing_dimensions) > (0.8 * diam):
#         return 1
#     else:
#         return 0

# def large_enough_area(lv, dim):
#     fairing_dimensions = lv.dimensions
#     area = dim[1]
#     if area is None:
#         return 0
#     if (fairing_dimensions[0] * fairing_dimensions[1]) > (0.65 * area):
#         return 1
#     else:
#         return 0

def get_performance(lv, lvInd, typ, h, i):
    coeffs = lv.loc[lvInd,typ]
    coeffsList = np.array(coeffs.replace('[','').replace(']','').split(','), dtype=float)
    hList = np.array([1, h, h**2], dtype=float)
    perf = np.dot(coeffsList, hList) # figure out performance coeffs
    return perf

def sufficient_performance(lv, m, typ, h, i):
    margin = 1.1  # 10% margin
    perf = get_performance(lv, typ, h, i)
    return perf > (margin * m)

def compute_number_of_launches(lv, lvInd, dims, numPlanes, numSats, wetMass, orb, a, e, i, rad):
    N = numPlanes * numSats
    perf = get_performance(lv, lvInd, orb, rToh(a,rad)/1000, i)
    if perf > 0:
        NL_mass = np.ceil((wetMass * N) / perf)
    else:
        NL_mass = 1e10
    NL_vol = np.ceil((np.prod(dims) * N) / (1/4*np.pi*lv.loc[lvInd,'diameter']**2 * lv.loc[lvInd,'height']))
    NL_diam = np.ceil((max(dims) * N) / max(lv.loc[lvInd,'diameter'], lv.loc[lvInd,'height']))
    num_launches = max(NL_mass, NL_vol, NL_diam)
    return num_launches

def compute_launch_cost(lv, lvInd, numLaunches, numPlanes, numSats):
    if numLaunches > (numPlanes * numSats):
        ccost = 1e10
    else:
        ccost = lv.loc[lvInd,'cost']
    launch_cost = numLaunches * ccost
    return launch_cost

def designLV(mission, spacecraft, compInstance, LVData):
    """
    Determines the launch vehicle to use based on the mass of the spacecraft
    """
    # pull params
    numPlanes = mission.numPlanes
    numSats = mission.numSats
    wetMass = spacecraft.wetMass
    orbit = mission.orbitType
    a = mission.a
    e = mission.e
    i = mission.i
    dims = spacecraft.dimensions
    rad = mission.rad

    # determine orbit type (for performance coefficients)
    if orbit in ["LEO", "SSO", "MEO", "GEO"]:
        orbitType = "payload-" + orbit
        if orbit == "LEO" and i < 45:
            orbitType = orbitType + "-equat"
        elif orbit == "LEO" and i >= 45:
            orbitType = orbitType + "-polar"
    else:
        raise ValueError("Invalid orbit type. Please choose from LEO, SSO, MEO, or GEO.")
    
    # initialize variables
    minLaunchCost = -1

    # call Functions
    for lvInd in range(len(LVData)):
        numLaunches = compute_number_of_launches(LVData, lvInd, dims, numPlanes, numSats, wetMass, orbitType, a, e, i, rad)
        launchCost = compute_launch_cost(LVData, lvInd, numLaunches, numPlanes, numSats)
        if launchCost < minLaunchCost or minLaunchCost == -1:
            minLaunchCost = launchCost
            bestLV = lvInd

    launchVehicle = compInstance
    launchVehicle.type = 'Launch Vehicle'
    launchVehicle.name = LVData.loc[bestLV,'id']
    launchVehicle.height = LVData.loc[bestLV,'height']
    launchVehicle.diameter = LVData.loc[bestLV,'diameter']
    launchVehicle.cost = minLaunchCost
    spacecraft.launchVehicle = launchVehicle
    return launchVehicle

