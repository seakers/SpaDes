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
maxCostList = maxCostComps(componentList)

# Optimize
numRuns = 20

# Genetic Algorithm
# t00 = time.time()
allHVGA = []
allMaxHVGA = []
for runGA in range(numRuns):
    print("RUN: ", runGA, "\n\n")
    allLocsGA, allDimsGA, numStepsGA, maxHyperVolumeGA, allHyperVolumeGA = optimization(componentList,maxCostList,"GA")
    allMaxHVGA.append(maxHyperVolumeGA)
    allHVGA.append(allHyperVolumeGA)
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

allMaxHVGA = np.array(allMaxHVGA)
# allMaxHVRL = np.array(allMaxHVRL)
bigMaxHVGA = np.max(allMaxHVGA,axis=0)
# bigMaxHVRL = np.max(allMaxHVRL,axis=0)
allHVGA = np.array(allHVGA)
# allHVRL = np.array(allHVRL)
meanHVGA = np.mean(allHVGA,0)
# meanHVRL = np.mean(allHVRL,0)
stdHVGA = np.std(allHVGA,0)
# stdHVRL = np.std(allHVRL,0)

plt.plot(meanHVGA,color='tab:blue')
# plt.plot(meanHVRL,color='tab:orange')
# plt.plot(bigMaxHVGA,color='tab:blue',linestyle='dashed')
# plt.plot(bigMaxHVRL,color='tab:orange',linestyle='dashed')
plt.fill_between(range(len(meanHVGA)), meanHVGA + stdHVGA, meanHVGA - stdHVGA, alpha=.5, linewidth=0, color='tab:blue')
# plt.fill_between(range(len(meanHVRL)), meanHVRL + stdHVRL, meanHVRL - stdHVRL, alpha=.5, linewidth=0, color='tab:orange')
# plt.legend(["Average Hypervolume Genetic Algorithm","Average Hypervolume Deep RL",
#             "Maximum Hypervolume Genetic Algorithm","Maximum Hypervolume Deep RL"],loc="upper left")
plt.legend(["Average Hypervolume Genetic Algorithm"],loc="upper left")
plt.xlabel("Number of Function Evaluations")
plt.ylabel("Hypervolume")
plt.title("Deep RL / Genetic Algorithm Hypervolume Comparison")
plt.show()

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
for i in range(len(componentList)):
    # x,y,z = get_cube(component.dimensions,component.location)
    xGA,yGA,zGA = getCube(allDimsGA[-1][i],allLocsGA[-1][i])
    plot1 = ax1.plot_surface(xGA,yGA,zGA)
plt.title("Visualization of Configuration GA")


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