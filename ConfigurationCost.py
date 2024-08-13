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

def wireCost(dimensions,locations,types,orientations):
    # Extra cost from wires from here https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Customer+Drawing%7F10614%7FK%7Fpdf%7FEnglish%7FENG_CD_10614_K.pdf%7F865042-004
    # using manhattan distance for each component from pcu
    # setting the ports to be in the center of the -x face of each component
    portDirBase = np.array([-1,0,0])
    PCUInd = types.index("PCU")
    PCULoc = locations[PCUInd]
    PCUPortDir = np.matmul(orientations[PCUInd],portDirBase)
    PCUPortLoc = np.array(PCULoc) + np.multiply(np.array(dimensions[PCUInd])/2,PCUPortDir)
    totWireLen = 0
    for ind,loc in enumerate(locations):
        portDir = np.matmul(orientations[ind],portDirBase)
        compPortLoc = np.array(loc) + np.multiply(np.array(dimensions[ind])/2,portDir)
        wireLen = np.abs(PCUPortLoc[0]-compPortLoc[0]) + np.abs(PCUPortLoc[1]-compPortLoc[1]) + np.abs(PCUPortLoc[2]-compPortLoc[2])
        totWireLen += wireLen

    return totWireLen

def thermalCost(dimensions,locations,heatDisps):
    # Assumes just radiation heat transfer between components
    # Score is the variance of Qnet bc we want uniform heat distribution
    Qin = np.zeros(len(locations))

    SA = np.zeros(len(dimensions))
    for ind,dims in enumerate(dimensions):
        SA[ind] = 2*(dims[0]*dims[1] + dims[0]*dims[2] + dims[1]*dims[2])

    compPairList = itertools.combinations(range(len(locations)),2)
    for pair in compPairList:
        locA = locations[pair[0]]
        locB = locations[pair[1]]
        SAA = SA[pair[0]]
        SAB = SA[pair[1]]
        heatDispsA = heatDisps[pair[0]]
        heatDispsB = heatDisps[pair[1]]
        r = np.sqrt((locA[0]-locB[0])**2 + (locA[1]-locB[1])**2 + (locA[2]-locB[2])**2)
        distanceSphere = 4*np.pi*r**2
        # calculate the Heat transfer. Assumes constant heat production (questionable assumption)
        # 1/4 SA is the average cross sectional area of any 3d shape. 4*pi*r^2 is the surface area of the sphere
        # of radius r from the dissipating component
        if SAA < distanceSphere:
            Qin[pair[0]] += 1/4*SAA*heatDispsB/(distanceSphere)
        else:
            Qin[pair[0]] += heatDispsB

        if SAB < distanceSphere:
            Qin[pair[1]] += 1/4*SAB*heatDispsA/(distanceSphere)
        else:
            Qin[pair[1]] += heatDispsA


    Qnet = Qin - np.array(heatDisps)
    QnetVar = np.var(Qnet)

    return QnetVar

def vibrationsCost(dimensions,locations):
    
    return 1

def maxEstimatedCosts(dimensions,masses,types,heatDisps,orientations):
    maxOverlapCost = overlapCost(dimensions,[[0,0,0]]*len(dimensions)) # all comps in center

    maxCmCost = centerMassCost(dimensions,[[1,1,1]]*len(dimensions),masses) # all comps in corner

    maxOffAxisInertia,maxOnAxisInertia = inertiaCost(dimensions,[[1,1,1]]*len(dimensions),masses) # all comps in corner

    wireLocs = [[1,1,1]]*len(dimensions)
    PCUInd = types.index("PCU")
    wireLocs[PCUInd] = [-1,-1,-1]
    maxWireCost = wireCost(dimensions,wireLocs,types,orientations) # all comps in opposite corner of PCU

    thermLocs = [[1,1,1]]*len(dimensions)
    altCorners = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]]
    sortHeats = sorted(heatDisps)
    heatCutoff = sortHeats[6]
    for i,heat in enumerate(heatDisps):
        if heat <= heatCutoff:
            thermLocs[i] = altCorners.pop()
    maxThermalCost = thermalCost(dimensions,thermLocs,heatDisps) # all comps in center

    return [maxOverlapCost,maxCmCost,maxOffAxisInertia,maxOnAxisInertia,maxWireCost,maxThermalCost]

def maxCostComps(components):
    # pull parameters from the components
    dimensions = []
    types = []
    masses = []
    heatDisps = []
    for comp in components:
        dimensions.append(np.matmul(np.abs(comp.orientation),comp.dimensions))
        types.append(comp.type)
        masses.append(comp.mass)
        heatDisps.append(comp.heatDisp)

    orientations = [np.array([[1,0,0],[0,1,0],[0,0,1]])]*len(components)

    maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal = maxEstimatedCosts(dimensions,masses,types,heatDisps,orientations)

    return [maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal]

def getCostComps(components, maxCosts):
    # pull parameters from the components
    locations = []
    dimensions = []
    orientations = []
    types = []
    masses = []
    heatDisps = []
    for comp in components:
        locations.append(comp.location)
        orientations.append(comp.orientation)
        dimensions.append(np.matmul(np.abs(comp.orientation),comp.dimensions))
        types.append(comp.type)
        masses.append(comp.mass)
        heatDisps.append(comp.heatDisp)

    # Get the cost from each cost source
    overlapCostVal = overlapCost(dimensions,locations)
    cmCostCalVal = centerMassCost(dimensions,locations,masses)
    offAxisInertia,onAxisInertia = inertiaCost(dimensions,locations,masses)
    wireCostVal = wireCost(dimensions,locations,types,orientations)
    thermalCostVal = thermalCost(dimensions,locations,heatDisps)

    # maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire = maxEstimatedCosts(dimensions,masses,types,heatDisps)
    maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal = maxCosts

    costList = [overlapCostVal/maxOverlap*100, cmCostCalVal/maxCM, offAxisInertia/maxOffAx, onAxisInertia/maxOnAx, wireCostVal/maxWire, thermalCostVal/maxThermal]
    return costList
