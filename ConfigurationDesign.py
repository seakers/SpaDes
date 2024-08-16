import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ConfigurationOptimization import *
from SCDesignClasses import Component
from ConfigurationCost import maxCostComps

def getCube(dimensions,location):
    # Function to transform cube dimensions and location into a form that plot_surface can plot
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta) * dimensions[0] + location[0]
    y = np.sin(Phi)*np.sin(Theta) * dimensions[1] + location[1]
    z = np.cos(Theta)/np.sqrt(2) * dimensions[2] + location[2]
    return x,y,z

# def updatePlot(num,allDimensions,allLocations):
#     # Function to update the plot for animation

#     # Clear the plot
#     ax2.cla()

#     # Creating and plotting elements
#     for i in range(len(allDimensions[0])):
#         x,y,z = getCube(allDimensions[num][i],allLocations[num][i])
#         ax2.plot_surface(x, y, z)

#     # Readjust plot
#     ax2.set_xlim(-1,1)
#     ax2.set_ylim(-1,1)
#     ax2.set_zlim(-1,1)
#     ax2.set_aspect('equal')

# Create Components to put in spacecraft. Same as ones used in 
# Spacecraft Component Adaptive Layout Environment (SCALE): An efficient optimization tool
# by Fakoor

componentList = [
    Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2),
    Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    Component(type="transmitter", mass=4.5, dimensions=[.25,.15,.05], heatDisp=12),
    Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10),
    Component(type="reciever", mass=4, dimensions=[.2,.15,.03], heatDisp=11),
    Component(type="reciever", mass=3, dimensions=[.175,.125,.03], heatDisp=9),
    Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7),
    Component(type="OBDH", mass=9, dimensions=[.24,.18,.18], heatDisp=6),
    Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
]

structPanelList = [
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,.5], orientation=getOrientation(0)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,-.5], orientation=getOrientation(1)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,.5,0], orientation=getOrientation(22)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,-.5,0], orientation=getOrientation(23)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[.5,0,0], orientation=getOrientation(16)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[-.5,0,0], orientation=getOrientation(18)),
] # orientations chosen manually so that the positive z normal (on dimensions array) is facing outwards. Components are placed on the +- z face of the panels
  # surfaceNormal is the normal of the face that the component is placed on, relative to the dimensions, not the orientation (aka will be changed by the orientation)

numComps = len(componentList)
compLocs = np.ndarray.tolist(np.random.normal(0,0.4,(numComps,3)))
compDims = []
i = 0
for comp in componentList:
    comp.location = compLocs[i]
    comp.orientation = np.eye(3)
    compDims.append(comp.dimensions)
    i+=1

# calculate here so it is only done once (for speed)
maxCostList = maxCostComps(componentList,structPanelList)

# Optimize
numRuns = 20

# Genetic Algorithm
# t00 = time.time()
allHVGA = []
for runGA in range(numRuns):
    print("\n\n\nRUN: ", runGA, "\n\n")
    numStepsGA, allHVGARun, pfSolutionsGA, pfCostsGA = optimization(componentList,structPanelList,maxCostList,"GA")
    allHVGA.append(allHVGARun)
# t01 = time.time()
# print("Time for RL2: ", (t01-t00)/numRuns)

# Reset locations and dimensions for Reinforcement Learning
# i = 0
# for comp in componentList:
#     comp.location = compLocs[i]
#     comp.dimensions = compDims[i]
#     i+=1
# # t10 = time.time()
# allHVRL = []
# allMaxHVRL = []
# for runRL in range(numRuns):
#     print("RUN: ", runRL, "\n\n")
#     allLocsRL, allDimsRL, numStepsRL, maxHyperVolumeRL, allHyperVolumeRL = optimization(componentList,maxCostList,"RL")
#     allMaxHVRL.append(maxHyperVolumeRL)
#     allHVRL.append(allHyperVolumeRL)
# t11 = time.time()
# print("Time for RLE: ", (t11-t10)/numRuns)

allHVGA = np.array(allHVGA)
# allHVRL = np.array(allHVRL)
medianHVGA = np.median(allHVGA,0)
# meanHVRL = np.mean(allHVRL,0)
q1HVGA = np.quantile(allHVGA,.25,axis=0)
q3HVGA = np.quantile(allHVGA,.75,axis=0)
maxHVGA = np.max(allHVGA,0)
minHVGA = np.min(allHVGA,0)
# stdHVRL = np.std(allHVRL,0)

plt.plot(medianHVGA,color='tab:blue')
# plt.plot(meanHVRL,color='tab:orange')
plt.plot(maxHVGA,color='tab:blue',linestyle='dashed')
plt.plot(minHVGA,color='tab:blue',linestyle='dashed')
# plt.plot(bigMaxHVRL,color='tab:orange',linestyle='dashed')
plt.fill_between(range(len(medianHVGA)), q1HVGA, q3HVGA, alpha=.5, linewidth=0, color='tab:blue')
# plt.fill_between(range(len(meanHVRL)), meanHVRL + stdHVRL, meanHVRL - stdHVRL, alpha=.5, linewidth=0, color='tab:orange')
# plt.legend(["Average Hypervolume Genetic Algorithm","Average Hypervolume Deep RL",
#             "Maximum Hypervolume Genetic Algorithm","Maximum Hypervolume Deep RL"],loc="upper left")
plt.legend(["Median Hypervolume GA","Maximum Hypervolume GA","Minimum Hypervolume GA"],loc="upper left")
plt.xlabel("Number of Function Evaluations")
plt.ylabel("Hypervolume")
plt.title("Deep RL / Genetic Algorithm Hypervolume Comparison")
plt.show()

# configure last pf solution into dims and locs to visualize
allDimsGA = []
allLocsGA = []
allTypesGA = []
solution = pfSolutionsGA[-1]
surfNormal = np.array([0,0,1])
for i in range(len(componentList)):

    allTypesGA.append(componentList[i].type)

    transMat = getOrientation(int(solution[4*i+3]))
    orientation = transMat

    panelChoice = structPanelList[int(solution[4*i]%len(structPanelList))]
    if solution[4*i] >= len(structPanelList):
        surfNormal = surfNormal * -1
    
    surfLoc = np.matmul(panelChoice.orientation,np.multiply([solution[4*i+1],solution[4*i+2],surfNormal[2]],np.array(panelChoice.dimensions)/2))
    allLocsGA.append(surfLoc + np.multiply(np.abs(np.matmul(transMat,np.array(componentList[i].dimensions)/2)),np.matmul(panelChoice.orientation,surfNormal)) + panelChoice.location)

    allDimsGA.append(np.matmul(transMat,componentList[i].dimensions))

panelDims = []
panelLocs = []
for panel in structPanelList:
    panelLocs.append(panel.location)
    panelDims.append(np.matmul(panel.orientation,panel.dimensions))


# Create Figure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

# Plot Adjustment
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)
ax1.set_aspect('equal')

# for i in range(numElements):
#     x,y,z = get_cube(elDims[i],elLocs[i])
#     plot = ax1.plot_surface(x, y, z)
proxyPoints = []
for i in range(len(componentList)):
    xGA,yGA,zGA = getCube(allDimsGA[i],allLocsGA[i])
    objColor = tuple(np.random.rand(3))
    plot1 = ax1.plot_surface(xGA,yGA,zGA, color=objColor, label=allTypesGA[i])
    point = ax1.scatter(allLocsGA[i][0],allLocsGA[i][1],allLocsGA[i][2],color=objColor)
    proxyPoints.append(point)


for j in range(len(structPanelList)):
    xPanel,yPanel,zPanel = getCube(panelDims[j],panelLocs[j])
    plot1 = ax1.plot_surface(xPanel,yPanel,zPanel, alpha=0.1, color='tab:gray')

plt.title("Visualization of Configuration GA")
plt.legend(proxyPoints,allTypesGA)


# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# ax2.set_xlim(-1,1)
# ax2.set_ylim(-1,1)
# ax2.set_zlim(-1,1)
# ax2.set_aspect('equal')

# for i in range(len(componentList)):
#     # x,y,z = get_cube(component.dimensions,component.location)
#     xRL,yRL,zRL = getCube(allDimsRL[-1][i],allLocsRL[-1][i])
#     plot2 = ax2.plot_surface(xRL,yRL,zRL)
# plt.title("Visualization of Configuration Deep RL")

# if numStepsGA > 0:
#     ani = animation.FuncAnimation(fig2, updatePlot, numStepsGA, fargs=(allDimsGA, allLocsGA))

# ax1.title("Visualization of Configuration GA")
# ax2.title("Visualization of Configuration GA Animation")
plt.show()