import numpy as np
import itertools
from ConfigUtils import getOrientation
import torch

def centerMassCost(locations,masses):
    # Function to calculate the center of mass
    locx = []
    locy = []
    locz = []
    for i in range(len(locations)):
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

def overlapCostSingleNP(components, design, structPanels):
    # Function to find if a new element overlaps with any existing elements
    # used for immediate reward in RL

    surfNormal = np.array([0,0,1])

    # pull parameters from the components
    numComps = int(len(design)/4)
    locations = []
    dimensions = []

    for panel in structPanels:
        locations.append(panel.location)
        dimensions.append(panel.dimensions)

    for i in range(numComps):

        transMat = getOrientation(int(design[4*i+3]))
    
        panelChoice = structPanels[int(design[4*i]%len(structPanels))]
        if design[4*i] >= len(structPanels):
            surfNormal = surfNormal * -1
        
        surfLoc = np.matmul(panelChoice.orientation,np.multiply([design[4*i+1],design[4*i+2],surfNormal[2]],np.array(panelChoice.dimensions)/2))
        locations.append(surfLoc + np.multiply(np.abs(np.matmul(transMat,np.array(components[i].dimensions)/2)),np.matmul(panelChoice.orientation,surfNormal)) + panelChoice.location)
        dimensions.append(components[i].dimensions)

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
    
    # Find the overlap between the last element and all other elements
    corners2 = elementCorners[-1]
    for corners1 in elementCorners[:-1]:
        xOverlap = min([corners1[1][0],corners2[1][0]]) - max([corners1[0][0],corners2[0][0]])
        yOverlap = min([corners1[1][1],corners2[1][1]]) - max([corners1[0][1],corners2[0][1]])
        zOverlap = min([corners1[1][2],corners2[1][2]]) - max([corners1[0][2],corners2[0][2]])
        if xOverlap >= 0 and yOverlap >= 0 and zOverlap >= 0:
            overlap += xOverlap*yOverlap*zOverlap

    return overlap

def overlapCostSingle(components, design, structPanels):
    surfNormal = torch.tensor([0, 0, 1], device='cuda', dtype=torch.float32)  # Ensure surfNormal is float32
    overlapAll = torch.zeros(len(design), device='cuda', dtype=torch.float32)

    numComps = int(len(design[0]) / 4)

    # Precompute structural panel locations, dimensions, and orientations as float32 tensors
    panel_locations = torch.stack([torch.tensor(panel.location, device='cuda', dtype=torch.float32) for panel in structPanels])
    panel_dimensions = torch.stack([torch.tensor(panel.dimensions, device='cuda', dtype=torch.float32) for panel in structPanels])
    panel_orientations = torch.stack([torch.tensor(panel.orientation, device='cuda', dtype=torch.float32) for panel in structPanels])

    for batchIdx, batchDes in enumerate(design):
        batchLocations = []
        batchDimensions = []
        for i in range(numComps):
            transMat = torch.tensor(getOrientation(int(batchDes[4 * i + 3])), device='cuda', dtype=torch.float32)

            # Precompute structural panel choice and normal adjustment
            panelIdx = int(batchDes[4 * i] % len(structPanels))
            orientation = panel_orientations[panelIdx]
            dimensions = panel_dimensions[panelIdx]

            surfNormal_adjusted = surfNormal if batchDes[4 * i] < len(structPanels) else surfNormal * -1
            
            surfLoc = torch.matmul(orientation, 
                                   (torch.tensor([batchDes[4 * i + 1], batchDes[4 * i + 2], surfNormal_adjusted[2]], device='cuda', dtype=torch.float32) * (dimensions / 2)))

            component_dim = torch.tensor(components[i].dimensions, device='cuda', dtype=torch.float32)
            transMat_component = torch.matmul(transMat, component_dim / 2)
            finalLoc = surfLoc + torch.abs(torch.matmul(transMat_component, surfNormal_adjusted)) + panel_locations[panelIdx]

            batchLocations.append(finalLoc)
            batchDimensions.append(component_dim)

        batchLocations = torch.stack(batchLocations)
        batchDimensions = torch.stack(batchDimensions)

        minCorners = batchLocations - batchDimensions / 2
        maxCorners = batchLocations + batchDimensions / 2

        # Vectorized overlap calculation
        corners2_min = minCorners[-1]
        corners2_max = maxCorners[-1]
        corners1_min = minCorners[:-1]
        corners1_max = maxCorners[:-1]

        xOverlap = torch.minimum(corners1_max[:, 0], corners2_max[0]) - torch.maximum(corners1_min[:, 0], corners2_min[0])
        yOverlap = torch.minimum(corners1_max[:, 1], corners2_max[1]) - torch.maximum(corners1_min[:, 1], corners2_min[1])
        zOverlap = torch.minimum(corners1_max[:, 2], corners2_max[2]) - torch.maximum(corners1_min[:, 2], corners2_min[2])

        overlap = torch.sum(torch.logical_and(torch.logical_and(xOverlap >= 0, yOverlap >= 0), zOverlap >= 0) * xOverlap * yOverlap * zOverlap)

        overlapAll[batchIdx] = overlap

    return overlapAll


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
            Qin[pair[0]] += 1/4*SAA*heatDispsB/(distanceSphere) # look into changing to r^2
        else:
            Qin[pair[0]] += heatDispsB

        if SAB < distanceSphere:
            Qin[pair[1]] += 1/4*SAB*heatDispsA/(distanceSphere) # look into changing to r^2
        else:
            Qin[pair[1]] += heatDispsA


    Qnet = Qin - np.array(heatDisps)
    QnetVar = np.var(Qnet)

    return QnetVar

def pointingObj(dimensions,locations,types,orientations):
    """
    Function to calculate if the pointing of certain components is clear
    """

    # setting pointing to be in the +x direction
    pointDir = np.array([1,0,0])

    pointingComps = ["solar panel","payload","transmitter","receiver","antenna","star tracker","sun sensor"]

    newPointLocs = []
    newPointDims = []

    for ind,comp in enumerate(types):
        if comp in pointingComps:
            compPointDir = np.matmul(orientations[ind],pointDir)
            pointLoc = np.array(locations[ind]) + np.multiply(np.array(dimensions[ind])/2,compPointDir)
            pointDimMain = np.multiply(compPointDir - pointLoc, compPointDir)
            pointDimOff = np.abs(np.matmul(orientations[ind],np.array([0,0.01,0.01])))
            pointDims = pointDimMain + pointDimOff
            pointLocCenter = pointLoc + np.multiply(pointDimMain/2,compPointDir)
            newPointLocs.append(pointLocCenter)
            newPointDims.append(pointDims)
        
    return newPointLocs,newPointDims

            
def constraintCost(dimensions,locations,types,orientations):
    """
    Function to calculate the cost of overlap and pointing of components
    First creates imaginary pointing componets, then calls the overlap function
    """

    newPointLocs,newPointDims = pointingObj(dimensions,locations,types,orientations)

    constrCost = overlapCost(dimensions + newPointDims,locations + newPointLocs)

    return constrCost


def vibrationsCost(dimensions,locations):
    
    return 1

def maxEstimatedCosts(dimensions,masses,types,heatDisps,orientations,structMasses,structDims,structLocs):
    maxConstraintCost = constraintCost(dimensions + structDims, [[0,0,0]]*len(dimensions) + structLocs, types, orientations) # all comps in center

    maxCmCost = centerMassCost([[1,1,1]]*len(dimensions) + structLocs, masses + structMasses) # all comps in corner

    maxOffAxisInertia,maxOnAxisInertia = inertiaCost(dimensions + structDims,[[1,1,1]]*len(dimensions) + structLocs, masses + structMasses) # all comps in corner

    wireLocs = [[1,1,1]]*len(dimensions)
    PCUInd = types.index("PCU")
    wireLocs[PCUInd] = [-1,-1,-1]
    maxWireCost = wireCost(dimensions,wireLocs,types,orientations) # all comps in opposite corner of PCU

    thermLocs = [[1,1,1]]*len(dimensions)
    altCorners = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]]
    sortHeats = sorted(heatDisps)
    heatCutoff = sortHeats[6]
    for i,heat in enumerate(heatDisps):
        if heat <= heatCutoff and altCorners:
            thermLocs[i] = altCorners.pop()
    maxThermalCost = thermalCost(dimensions,thermLocs,heatDisps) # all comps in center

    return [maxConstraintCost,maxCmCost,maxOffAxisInertia,maxOnAxisInertia,maxWireCost,maxThermalCost]

def maxCostComps(components, structPanels):
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

    structMasses = []
    structDims = []
    structLocs = []
    for panel in structPanels:
        structMasses.append(panel.mass)
        structDims.append(np.matmul(np.abs(panel.orientation),panel.dimensions))
        structLocs.append(panel.location)

    orientations = [np.array([[1,0,0],[0,1,0],[0,0,1]])]*len(components)

    maxConstraint,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal = maxEstimatedCosts(dimensions,masses,types,heatDisps,orientations,structMasses,structDims,structLocs)

    return [maxConstraint,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal]

def getCostComps(components, structPanels, maxCosts):
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

    structMasses = []
    structDims = []
    structLocs = []
    for panel in structPanels:
        structMasses.append(panel.mass)
        structDims.append(np.matmul(np.abs(panel.orientation),panel.dimensions))
        structLocs.append(panel.location)

    # Get the cost from each cost source
    constraintCostVal = constraintCost(dimensions + structDims, locations + structLocs, types, orientations)
    cmCostCalVal = centerMassCost(locations + structLocs, masses + structMasses)
    offAxisInertia,onAxisInertia = inertiaCost(dimensions + structDims, locations + structLocs, masses + structMasses)
    wireCostVal = wireCost(dimensions,locations,types,orientations)
    thermalCostVal = thermalCost(dimensions,locations,heatDisps)

    # maxOverlap,maxCM,maxOffAx,maxOnAx,maxWire = maxEstimatedCosts(dimensions,masses,types,heatDisps)
    maxConstraint,maxCM,maxOffAx,maxOnAx,maxWire,maxThermal = maxCosts

    costList = [constraintCostVal/maxConstraint, cmCostCalVal/maxCM, offAxisInertia/maxOffAx, onAxisInertia/maxOnAx, wireCostVal/maxWire, thermalCostVal/maxThermal]
    return costList
