from SCDesignClasses import Component
from ConfigUtils import getOrientation
import matplotlib.pyplot as plt
import numpy as np


def getCube(dimensions,location):
    # Function to transform cube dimensions and location into a form that plot_surface can plot
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta) * dimensions[0] + location[0]
    y = np.sin(Phi)*np.sin(Theta) * dimensions[1] + location[1]
    z = np.cos(Theta)/np.sqrt(2) * dimensions[2] + location[2]
    return x,y,z

exComp = Component(type="battery", mass=8, dimensions=[0.1,0.2,0.3], heatDisp=2)

# electrical ports are located on the -x side and pointing is located on the +x side, as the dimensions are defined

structPanelList = [
    # Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,.5], orientation=getOrientation(0)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,-.5], orientation=getOrientation(1)),
    # Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,.5,0], orientation=getOrientation(22)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,-.5,0], orientation=getOrientation(23)),
    # Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[.5,0,0], orientation=getOrientation(16)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[-.5,0,0], orientation=getOrientation(18)),
]

exSol = [4,.6,.6,15]

surfNormal = np.array([0, 0, 1])

type = exComp.type

transMat = getOrientation(int(exSol[3]))

panelChoice = structPanelList[int(exSol[0] % len(structPanelList))]
if exSol[0] >= len(structPanelList):
    surfNormal = surfNormal * -1

surfLoc = np.matmul(panelChoice.orientation, np.multiply([exSol[1], exSol[2], surfNormal[2]], np.array(panelChoice.dimensions) / 2))
locs = surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(exComp.dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location

dims = np.matmul(transMat, exComp.dimensions)

panelDims = []
panelLocs = []
for panel in structPanelList:
    panelLocs.append(panel.location)
    panelDims.append(np.matmul(panel.orientation, panel.dimensions))

# Create Figure for GA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Adjustment for GA
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_aspect('equal')

proxyPointsGA = []

xGA, yGA, zGA = getCube(dims, locs)
ax.plot_surface(xGA, yGA, zGA, color='red', label=type)
point = ax.scatter(locs[0], locs[1], locs[2], color='red')
proxyPointsGA.append(point)

for j in range(len(structPanelList)):
    xPanel, yPanel, zPanel = getCube(panelDims[j], panelLocs[j])
    ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

# Plot quivers for component coordinates
quiver_x = ax.quiver(locs[0], locs[1], locs[2], 0, -0.4, 0, color='black')
quiver_y = ax.quiver(locs[0], locs[1], locs[2], 0.4, 0, 0, color='black')
quiver_z = ax.quiver(locs[0], locs[1], locs[2], 0, 0, 0.4, color='black')

# Add labels for the component quivers
ax.text(locs[0], locs[1] - 0.5, locs[2], 'x', color='black')
ax.text(locs[0] + 0.5, locs[1], locs[2], 'y', color='black')
ax.text(locs[0], locs[1], locs[2] + 0.4, 'z', color='black')

# Plot quivers for body coordinates
quiver_x_body = ax.quiver(0, 0, 0, 0.5, 0, 0, color='blue')
quiver_y_body = ax.quiver(0, 0, 0, 0, 0.5, 0, color='blue')
quiver_z_body = ax.quiver(0, 0, 0, 0, 0, 0.5, color='blue')

# Add labels for the body quivers
ax.text(0.6, 0, 0, 'x', color='blue')
ax.text(0, 0.5, 0, 'y', color='blue')
ax.text(0, 0, 0.5, 'z', color='blue')

# Add legend
black_arrow = plt.Line2D([0], [0], linestyle="none", marker=">", markersize=10, markerfacecolor="black", markeredgecolor="black")
blue_arrow = plt.Line2D([0], [0], linestyle="none", marker=">", markersize=10, markerfacecolor="blue", markeredgecolor="blue")
ax.legend([black_arrow, blue_arrow], ['Component Coordinates', 'Body Coordinates'], numpoints=1)

plt.title("Visualization of Coordinates")

plt.show()