import numpy as np

def enumerateOrientations():
    # Function to enumerate all possible right angle orientations of a component

    coordsA = np.array([1,2,3])
    coordsB = np.array([2,-1,3])
    flipMult = np.array([-1,-1,1])
    matLine = np.array([1,0,0])

    allCoords = []
    allMats = []
    for iA in range(len(coordsA)):
        newCoordsA = np.roll(coordsA,iA)
        allCoords.append(newCoordsA)
        for jA in range(len(flipMult)):
            flipMultTemp = np.roll(flipMult,jA)
            newFlipCoordsA = np.multiply(newCoordsA,flipMultTemp)
            allCoords.append(newFlipCoordsA)
    
    for iB in range(len(coordsB)):
        newCoordsB = np.roll(coordsB,iB)
        allCoords.append(newCoordsB)
        for jB in range(len(flipMult)):
            flipMultTemp = np.roll(flipMult,jB)
            newFlipCoordsB = np.multiply(newCoordsB,flipMultTemp)
            allCoords.append(newFlipCoordsB)
    
    for iM in range(len(allCoords)):
        newMat = np.array([matLine,matLine,matLine])
        for jM in range(len(allCoords[iM])):
            newMat[jM] = np.roll(newMat[jM],abs(allCoords[iM][jM])-1)
            newMat[jM] = newMat[jM]*np.sign(allCoords[iM][jM])
        allMats.append(newMat)
    
    return allCoords, allMats

def getOrientation(orientationChoice):
    # function to get the transformation matrix for a given orientation choice

    coordsA = np.array([1,2,3])
    coordsB = np.array([2,-1,3])
    flipMult = np.array([-1,-1,1])
    matBase = np.array([[1,0,0],[1,0,0],[1,0,0]])

    # choose between even and odd orientations
    if orientationChoice >= 0 and orientationChoice <= 11:
        coordsUsed = coordsA
    elif orientationChoice >= 12 and orientationChoice <= 23:
        coordsUsed = coordsB

    rollNum = np.floor_divide(orientationChoice,4)
    coord = np.roll(coordsUsed,rollNum)
    if np.mod(orientationChoice,4) != 0:
        rollNumFlip = np.mod(orientationChoice,4)
        newFlipMult = np.roll(flipMult,rollNumFlip)
        coord = np.multiply(coord,newFlipMult)

    # create the transformation matrix
    transMat = matBase
    for i in range(3):
        transMatPos = np.roll(matBase[i],abs(coord[i])-1)
        transMat[i] = transMatPos*np.sign(coord[i])

    return transMat

# transMat = getOrientation(5)
# dims = np.array([1,2,3])
# newDims = np.matmul(np.abs(transMat),dims)
# print(transMat)
# print(newDims)
