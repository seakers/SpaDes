import numpy as np
import pickle
from ConfigUtils import *
import matplotlib.pyplot as plt
from SCDesignClasses import *
from ComponentList import getComponents

folder = "ResultGraphs/2025-03-05_20-48-47_50_Comp_150_epoch_32_batch_20_run"

pfronts = np.load(f"{folder}/pareto_fronts.npz", allow_pickle=True)

pfSolutionsGA = pfronts['pfSolutionsGA']
pfPointsGA = pfronts['pfPointsGA']
pfSolutionsRL = pfronts['pfSolutionsRL']
pfPointsRL = pfronts['pfPointsRL']
pfSolutionsRS = pfronts['pfSolutionsRS']
pfPointsRS = pfronts['pfPointsRS']


print("\nGA Filtered Pareto Front")
for idx,sol in enumerate(pfSolutionsGA):
    if len(sol[0]['Structure']) > 8:
        print("\nDesign:", sol, "\nCosts:", pfPointsGA[idx])

print("\nRL Filtered Pareto Front")
for idx,sol in enumerate(pfSolutionsRL):
    if len(sol[0]['Structure']) > 8:
        print("\nDesign:", sol, "\nCosts:", pfPointsRL[idx])

print("\nRS Filtered Pareto Front")
for idx,sol in enumerate(pfSolutionsRS):
    if len(sol[0]['Structure']) > 8:
        print("Found at index: ", idx)
        print("\nDesign:", sol, "\nCosts:", pfPointsRS[idx])

componentList, transferLearningComponents = getComponents()

solutionRSIdx = 10
solDictRS = pfSolutionsRS[solutionRSIdx]
numComps = len(componentList)

# for solutionRSIdx, solDictRS in enumerate(pfSolutionsRS):
allDimsCompsRS = []
allLocsCompsRS = []
allOrientsCompsRS = []
allTypesCompsRS = []
allDimsStructRS = []
allLocsStructRS = []
allOrientsStructRS = []

if isinstance(solDictRS, dict):
    structSolRS = solDictRS['Structure']
    compSolRS = solDictRS['Components']
else:
    structSolRS = solDictRS[0]['Structure']
    compSolRS = solDictRS[0]['Components']


structPanels = []
numPanels = int(len(structSolRS)/8)
for j in range(numPanels):
    panelSize = [structSolRS[8*j],structSolRS[8*j+1]]
    panelLoc = [structSolRS[8*j+2],structSolRS[8*j+3],structSolRS[8*j+4]]
    panelEAngles = [structSolRS[8*j+5],structSolRS[8*j+6],structSolRS[8*j+7]]
    panelDCM = Euler2DCM(panelEAngles)
    # print("Panel DCM: ", panelDCM)
    # allStructDCMs.append(panelDCM)

    newPanel = StructPanel(dimensions=panelSize, location=panelLoc, orientation=panelDCM)
    structPanels.append(newPanel)

    allDimsStructRS.append(panelSize)
    allLocsStructRS.append(panelLoc)
    allOrientsStructRS.append(panelDCM)

surfNormal = np.array([0,0,1])

for i in range(numComps):
    compPanel = compSolRS[5*i]
    compLoc = [compSolRS[5*i+1],compSolRS[5*i+2]]
    compFace = compSolRS[5*i+3]
    compRot = compSolRS[5*i+4]

    compOrient = faceAndRot2DCM(faceChoice=compFace, rot=compRot)

    if compPanel >= len(structPanels): # plus one because the first choice is the top of the lv adapter, and no comps can go on the bottom
        panelChoice = structPanels[int(compPanel - len(structPanels)) + 1]
        panelChoiceDCM = panelChoice.orientation
        panelChoiceDCM = np.matmul(panelChoiceDCM,np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
        surfNormal = surfNormal*-1
    else:
        panelChoice = structPanels[int(compPanel)]
        panelChoiceDCM = panelChoice.orientation
    
    compDCM = align_cuboid_with_plane(compOrient,panelChoiceDCM)
    # print("Component DCM: ", compDCM)
    componentList[i].orientation = compDCM
    allOrientsCompsRS.append(compDCM)

    if compFace%3 == 0:
        dimOffset = componentList[i].dimensions[2]/2
    elif compFace%3 == 1:
        dimOffset = componentList[i].dimensions[1]/2
    elif compFace%3 == 2:
        dimOffset = componentList[i].dimensions[0]/2

    panelOffset = panelChoice.thickness/2

    offsetVect = np.matmul(panelChoiceDCM,surfNormal*(dimOffset+panelOffset))
    
    surfLoc = np.matmul(panelChoiceDCM,np.multiply([compLoc[0],compLoc[1],surfNormal[2]],np.array(panelChoice.dimensions)/2))
    # compLoc = surfLoc + np.multiply(np.abs(np.matmul(compDCM,np.array(componentList[i].dimensions)/2)),np.matmul(panelChoiceDCM,surfNormal)) + panelChoice.location
    compLoc = surfLoc + offsetVect + panelChoice.location
    # print("surfLoc: ", surfLoc)
    # print("offsetVect: ", offsetVect)
    # print("panelLoc: ", panelChoice.location)

    # compLoc = surfLoc + panelChoice.location
    # print("CompLocGraph: ", compLoc)
    componentList[i].location = compLoc
    allLocsCompsRS.append(compLoc)
    allDimsCompsRS.append(componentList[i].dimensions)

    # print("\nComp DCM: ", componentList[i].orientation, "\nPanelSide: ", compPanel, "\nCompFace: ", compFace, "\nRotationAngle: ", compRot)


# Create Figure for RS
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Adjustment for RS
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_aspect('equal')

objColor = tuple(np.random.rand(len(componentList), 3))

proxyPointsRS = []
for i in range(len(componentList)):
    xRS, yRS, zRS = getCube(allDimsCompsRS[i], allLocsCompsRS[i], allOrientsCompsRS[i])
    # ax.plot_surface(xRS, yRS, zRS, color=objColor[i], label=allTypesCompsRS[i])
    ax.plot_surface(xRS, yRS, zRS)
    # point = ax.scatter(allLocsCompsRS[i][0], allLocsCompsRS[i][1], allLocsCompsRS[i][2], color=objColor[i])
    # proxyPointsRS.append(point)

for j in range(len(structPanels)):
    xPanel, yPanel, zPanel = getCube(allDimsStructRS[j], allLocsStructRS[j], allOrientsStructRS[j])
    ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

plt.title("Visualization of Configuration RS")
# plt.legend(proxyPointsRS, allTypesRS, loc='center left', bbox_to_anchor=(1.1, 0.5))

plt.savefig(f"{folder}/RSConfig_{solutionRSIdx}")
pickle.dump(fig, open(f"{folder}/RSConfig_{solutionRSIdx}Interactive", "wb"))
