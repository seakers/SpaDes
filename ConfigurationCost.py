import numpy as np
import itertools

def centerMassCost(dimensions,locations,masses):
    # Function to calculate the center of mass
    locx = []
    locy = []
    locz = []
    for i in range(len(dimensions)):
        # Separate x,y,z components
        locx.append(locations[i][0])
        locy.append(locations[i][1])
        locz.append(locations[i][2])

    masses = np.array(masses)
    locx = np.array(locx)
    locy = np.array(locy)
    locz = np.array(locz)
    cmx = sum(np.multiply(masses,locx))/sum(masses)
    cmy = sum(np.multiply(masses,locy))/sum(masses)
    cmz = sum(np.multiply(masses,locz))/sum(masses)
    cmMag = np.sqrt(cmx**2 + cmy**2 + cmz**2)
    return cmMag


def inertiaCost(dimensions,locations,masses):
    # Function to calculate the moment of inertia
    # Method here is very simplified, will use full matrix and more complex analysis later
    # Treats each component as a point mass and minimizes sum of the magnitude of each inertia
    inertia = np.zeros((3,3))
    for i in range(len(dimensions)):
        Icm = 1/12*masses[i]*np.array([[(dimensions[i][1]**2 + dimensions[i][2]**2),0,0],
                          [0,(dimensions[i][0]**2 + dimensions[i][2]**2),0],
                          [0,0,(dimensions[i][0]**2 + dimensions[i][1]**2)]])
        skewSym = np.array([[0,-locations[i][2],locations[i][1]],
                              [locations[i][2],0,-locations[i][0]],
                              [-locations[i][1],locations[i][0],0]])
        Ir = masses[i]*np.matmul(skewSym,skewSym)
        inertia += Icm - Ir
    
    # Sum the off-axis moments of inertia
    offAxisInertia = abs(inertia[0,1]) + abs(inertia[0,2]) + abs(inertia[1,2])
    onAxisInertia = inertia[0,0] + inertia[1,1] + inertia [2,2]
    return offAxisInertia, onAxisInertia

def overlapCost(dimensions,locations):
    # Function to find how much overlap there are in all the elements
    overlap = 0

    # Find the min(x,y,z) and max(x,y,z) for each element
    elementCorners = []
    for i in range(len(dimensions)):
        minCorner = [locations[i][0]-dimensions[i][0]/2,
                     locations[i][1]-dimensions[i][1]/2,
                     locations[i][2]-dimensions[i][2]/2]
        maxCorner = [locations[i][0]+dimensions[i][0]/2,
                     locations[i][1]+dimensions[i][1]/2,
                     locations[i][2]+dimensions[i][2]/2]
        elCorners = [minCorner,maxCorner]
        elementCorners.append(elCorners)
    
    # Find the overlap between each pair of elements
    elCombList = itertools.combinations(elementCorners,2)
    for comb in elCombList:
        (corners1,corners2) = comb
        xOverlap = min([corners1[1][0],corners2[1][0]]) - max([corners1[0][0],corners2[0][0]])
        yOverlap = min([corners1[1][1],corners2[1][1]]) - max([corners1[0][1],corners2[0][1]])
        zOverlap = min([corners1[1][2],corners2[1][2]]) - max([corners1[0][2],corners2[0][2]])
        if xOverlap >= 0 and yOverlap >= 0 and zOverlap >= 0:
            overlap += xOverlap*yOverlap*zOverlap

    return overlap

def wireCost(locations,types):
    # Extra cost from wires from here https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Customer+Drawing%7F10614%7FK%7Fpdf%7FEnglish%7FENG_CD_10614_K.pdf%7F865042-004
    # using manhattan distance for each component from pcu
    PCUInd = types.index("PCU")
    PCULoc = locations[PCUInd]
    totWireLen = 0
    for loc in locations:
        wireLen = np.abs(PCULoc[0]-loc[0]) + np.abs(PCULoc[1]-loc[1]) + np.abs(PCULoc[2]-loc[2])
        totWireLen += wireLen

    return totWireLen

def thermalCost(locations,heatDisps):
    # Assumes direct line of conduction for the heat transfer between components
    compPairList = itertools.combinations(range(len(heatDisps)))
    for pair in compPairList:
        x=1
    return 1

def vibrationsCost(dimensions,locations):
    
    return 1

def maxEstimatedCosts(dimensions,masses,types,heatDisps):
    maxOverlapCost = overlapCost(dimensions,[[0,0,0]]*len(dimensions)) # all comps in center
    maxCmCost = centerMassCost(dimensions,[[1,1,1]]*len(dimensions),masses) # all comps in corner
    maxOffAxisInertia,maxOnAxisInertia = inertiaCost(dimensions,[[1,1,1]]*len(dimensions),masses) # all comps in corner
    wireLocs = [[1,1,1]]*len(dimensions)
    PCUInd = types.index("PCU")
    wireLocs[PCUInd] = [-1,-1,-1]
    maxWireCost = wireCost(wireLocs,types) # all comps in opposite corner of PCU
    # maxThermalCost = thermalCost([[0,0,0]]*len(dimensions),heatDisps) # all comps in center
    # return [maxOverlapCost,maxCmCost,maxOffAxisInertia,maxOnAxisInertia,maxWireCost,maxThermalCost]
    return [maxOverlapCost,maxCmCost,maxOffAxisInertia,maxOnAxisInertia,maxWireCost]


# def getMaxCosts(components):
#     # pull parameters from the components
#     dimensions = []
#     types = []
#     masses = []
#     for comp in components:
#         dimensions.append(comp.dimensions)
#         types.append(comp.type)
#         masses.append(comp.mass)

#     return maxEstimatedCosts(dimensions,masses,types)

def getCostComps(components):
    # pull parameters from the components
    locations = []
    dimensions = []
    types = []
    masses = []
    heatDisps = []
    for comp in components:
        locations.append(comp.location)
        dimensions.append(comp.dimensions)
        types.append(comp.type)
        masses.append(comp.mass)
        heatDisps.append(comp.heatDisp)

    # Get the cost from each cost source
    overlapCostVal = overlapCost(dimensions,locations)
    cmCostCalVal = centerMassCost(dimensions,locations,masses)
    offAxisInertia,onAxisInertia = inertiaCost(dimensions,locations,masses)
    wireCostVal = wireCost(locations,types)
    # thermalCostVal = thermalCost(locations, heatDisps)

    maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire = maxEstimatedCosts(dimensions,masses,types,heatDisps)
    # maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal = maxEstimatedCosts(dimensions,masses,types,heatDisps)

    # Add all the costs together
    # In the future add weights for different costs
    # costList = [overlapCostVal*100, cmCostCalVal*10, offAxisInertia, onAxisInertia*0.1, wireCostVal]
    # costList = [overlapCostVal/maxOverlap*100, cmCostCalVal/maxCM, offAxisInertia/maxOffAx, onAxisInertia/maxOnAx, wireCostVal/maxWire, thermalCostVal/maxThermal]
    costList = [overlapCostVal/maxOverlap*100, cmCostCalVal/maxCM, offAxisInertia/maxOffAx, onAxisInertia/maxOnAx, wireCostVal/maxWire]
    return costList

# def getCostParams(dimensions,locations,types,masses):
#     # Get the cost from each cost source
#     # Only needed for gradient method (for some reason?)
#     overlapCostVal = overlapCost(dimensions,locations)
#     cmCostCalVal = centerMassCost(dimensions,locations,masses)
#     offAxisInertia,onAxisInertia = inertiaCost(dimensions,locations,masses)
#     wireCostVal = wireCost(locations,types)

#     # maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire = maxEstimatedCosts(dimensions,masses,types)

#     # Add all the costs together
#     # In the future add weights for different costs
#     costList = [overlapCostVal*100, cmCostCalVal*10, offAxisInertia, onAxisInertia*0.1, wireCostVal]
#     # costList = [overlapCostVal/maxOverlap, cmCostCalVal/maxCM, offAxisInertia/maxOffAx, onAxisInertia/maxOnAx, wireCostVal/maxWire]
#     return costList