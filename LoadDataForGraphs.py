import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ConfigurationOptimization import *
from SCDesignClasses import Component
from ConfigurationCost import maxCostComps
import datetime
import os

def getCube(dimensions,location):
    # Function to transform cube dimensions and location into a form that plot_surface can plot
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta) * dimensions[0] + location[0]
    y = np.sin(Phi)*np.sin(Theta) * dimensions[1] + location[1]
    z = np.cos(Theta)/np.sqrt(2) * dimensions[2] + location[2]
    return x,y,z

componentList = [
    Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5),
    Component(type="solar panel", mass=1.6, dimensions=[.21,.52,.01], heatDisp=1.6),
    Component(type="solar panel", mass=1.4, dimensions=[.19,.49,.01], heatDisp=1.4),

    Component(type="payload", mass=6.5, dimensions=[.3,.24,.22], heatDisp=3.2),
    Component(type="payload", mass=5.5, dimensions=[.28,.22,.2], heatDisp=3),

    Component(type="transmitter", mass=3.8, dimensions=[.25,.1,.08], heatDisp=12),
    Component(type="transmitter", mass=4.0, dimensions=[.23,.12,.09], heatDisp=11),

    Component(type="receiver", mass=3.3, dimensions=[.21,.12,.05], heatDisp=9),
    Component(type="receiver", mass=3.5, dimensions=[.22,.13,.06], heatDisp=9.5),

    Component(type="antenna", mass=4.5, dimensions=[.35,.14,.12], heatDisp=9.7),
    Component(type="antenna", mass=3.2, dimensions=[.24,.1,.08], heatDisp=8),
    Component(type="antenna", mass=4.0, dimensions=[.34,.15,.1], heatDisp=9),

    Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.3),
    Component(type="star tracker", mass=1.8, dimensions=[.12,.14,.11], heatDisp=1.4),
    Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.1], heatDisp=1.2),

    Component(type="sun sensor", mass=1.2, dimensions=[.1,.09,.08], heatDisp=0.9),
    Component(type="sun sensor", mass=1.1, dimensions=[.1,.08,.07], heatDisp=1),
    Component(type="sun sensor", mass=1.3, dimensions=[.12,.1,.09], heatDisp=1.1),

    Component(type="battery", mass=5.8, dimensions=[.23,.21,.13], heatDisp=2.2),
    Component(type="battery", mass=6.0, dimensions=[.24,.22,.14], heatDisp=2.4),

    Component(type="PCU", mass=7, dimensions=[.26,.2,.14], heatDisp=6.5),
    Component(type="PCU", mass=6.5, dimensions=[.25,.19,.13], heatDisp=6.3),

    Component(type="OBDH", mass=9, dimensions=[.24,.19,.16], heatDisp=5.8),
    Component(type="OBDH", mass=8.8, dimensions=[.23,.18,.15], heatDisp=5.7),

    Component(type="reaction wheel", mass=3, dimensions=[.14,.12,.1], heatDisp=3.2),
    Component(type="reaction wheel", mass=3.2, dimensions=[.15,.13,.11], heatDisp=3.3),

    Component(type="propellant tank", mass=13, dimensions=[.3,.25,.2], heatDisp=4.2),
    Component(type="propellant tank", mass=12.5, dimensions=[.29,.24,.19], heatDisp=4.1),

    Component(type="attitude thruster", mass=2.5, dimensions=[.15,.14,.12], heatDisp=4.8),
    Component(type="attitude thruster", mass=2.7, dimensions=[.16,.15,.13], heatDisp=5),

    Component(type="IMU", mass=2.5, dimensions=[.14,.12,.09], heatDisp=2.5),
    Component(type="IMU", mass=2.4, dimensions=[.13,.11,.08], heatDisp=2.4),

    Component(type="atomic clock", mass=1.8, dimensions=[.12,.11,.07], heatDisp=1.6),

    Component(type="heater", mass=1.2, dimensions=[.09,.07,.05], heatDisp=1.9),
    Component(type="heater", mass=1.3, dimensions=[.1,.08,.06], heatDisp=2),

    Component(type="gyro", mass=3.1, dimensions=[.19,.13,.02], heatDisp=2.8),
    Component(type="gyro", mass=3.0, dimensions=[.18,.12,.02], heatDisp=2.7),

    Component(type="magnetometer", mass=1.4, dimensions=[.1,.12,.1], heatDisp=1.1),
    Component(type="magnetometer", mass=1.5, dimensions=[.11,.13,.09], heatDisp=1.2),

    Component(type="accelerometer", mass=1.8, dimensions=[.13,.11,.09], heatDisp=2.2),
    Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.1)
]



# electrical ports are located on the -x side and pointing is located on the +x side, as the dimensions are defined

structPanelList = [
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,.5], orientation=getOrientation(0)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,0,-.5], orientation=getOrientation(1)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,.5,0], orientation=getOrientation(22)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[0,-.5,0], orientation=getOrientation(23)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[.5,0,0], orientation=getOrientation(16)),
    Component(type="structural panel", mass=1, dimensions=[1,1,.01], location=[-.5,0,0], orientation=getOrientation(18)),
] # orientations chosen manually so that the positive z normal (on dimensions array) is facing outwards. Components are placed on the +- z face of the panels
# surfaceNormal is the normal of the face that the component is placed on, relative to the dimensions, not the orientation (aka will be changed by the orientation)


graph_data = np.load('ResultGraphs/2024-11-27_15-14-16/graph_data.npz')
pareto_fronts = np.load('ResultGraphs/2024-11-27_12-34-13/pareto_fronts.npz')

date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f'ResultGraphs/{date_str}', exist_ok=True)

# hypervolume
medianHVGA = graph_data['medianHVGA']
q1HVGA = graph_data['q1HVGA']
q3HVGA = graph_data['q3HVGA']
maxHVGA = graph_data['maxHVGA']
minHVGA = graph_data['minHVGA']
medianHVRL = graph_data['medianHVRL']
q1HVRL = graph_data['q1HVRL']
q3HVRL = graph_data['q3HVRL']
maxHVRL = graph_data['maxHVRL']
minHVRL = graph_data['minHVRL']
medianHVRS = graph_data['medianHVRS']
q1HVRS = graph_data['q1HVRS']
q3HVRS = graph_data['q3HVRS']
maxHVRS = graph_data['maxHVRS']
minHVRS = graph_data['minHVRS']

# costs
medianAvgCostsGA = graph_data['medianAvgCostsGA']
q1AvgCostsGA = graph_data['q1AvgCostsGA']
q3AvgCostsGA = graph_data['q3AvgCostsGA']
maxAvgCostsGA = graph_data['maxAvgCostsGA']
minAvgCostsGA = graph_data['minAvgCostsGA']
medianAvgCostsRL = graph_data['medianAvgCostsRL']
q1AvgCostsRL = graph_data['q1AvgCostsRL']
q3AvgCostsRL = graph_data['q3AvgCostsRL']
maxAvgCostsRL = graph_data['maxAvgCostsRL']
minAvgCostsRL = graph_data['minAvgCostsRL']
medianAvgCostsRS = graph_data['medianAvgCostsRS']
q1AvgCostsRS = graph_data['q1AvgCostsRS']
q3AvgCostsRS = graph_data['q3AvgCostsRS']
maxAvgCostsRS = graph_data['maxAvgCostsRS']
minAvgCostsRS = graph_data['minAvgCostsRS']

# pareto fronts
pfSolutionsGA = pareto_fronts['pfSolutionsGA']
pfPointsGA = pareto_fronts['pfPointsGA']
pfSolutionsRL = pareto_fronts['pfSolutionsRL']
pfPointsRL = pareto_fronts['pfPointsRL']
pfSolutionsRS = pareto_fronts['pfSolutionsRS']
pfPointsRS = pareto_fronts['pfPointsRS']


plt.figure()
plt.plot(medianHVRL, color='tab:orange')
plt.plot(maxHVRL, color='tab:orange', linestyle='dashed')
plt.plot(minHVRL, color='tab:orange', linestyle='dotted')
plt.fill_between(range(len(medianHVRL)), q1HVRL, q3HVRL, alpha=.5, linewidth=0, color='tab:orange')

plt.plot(medianHVGA, color='tab:blue')
plt.plot(maxHVGA, color='tab:blue', linestyle='dashed')
plt.plot(minHVGA, color='tab:blue', linestyle='dotted')
plt.fill_between(range(len(medianHVGA)), q1HVGA, q3HVGA, alpha=.5, linewidth=0, color='tab:blue')

plt.plot(medianHVRS, color='tab:green')
plt.plot(maxHVRS, color='tab:green', linestyle='dashed')
plt.plot(minHVRS, color='tab:green', linestyle='dotted')
plt.fill_between(range(len(medianHVRS)), q1HVRS, q3HVRS, alpha=.5, linewidth=0, color='tab:green')

plt.legend(["Median Hypervolume Transformer", "Maximum Hypervolume Transformer", "Minimum Hypervolume Transformer", "Interquartile Hypervolume Transformer",
            "Median Hypervolume GA", "Maximum Hypervolume GA", "Minimum Hypervolume GA", "Interquartile Hypervolume GA",
            "Median Hypervolume RS", "Maximum Hypervolume RS", "Minimum Hypervolume RS", "Interquartile Hypervolume RS"], 
            loc='lower right', fontsize='small')
plt.ylim(.4, .7)
plt.yticks(np.arange(.4, .72, 0.02))
plt.xlabel("Number of Function Evaluations")
plt.ylabel("Hypervolume")
# plt.title("Transformer / Genetic Algorithm / Random Search Hypervolume Comparison")
plt.savefig(f"ResultGraphs/{date_str}/HypervolumeComparisonTransformer")

fig, axs = plt.subplots(3, 2, figsize=(12, 10))
cost_labels = ["Overlap", "Moment of Inertia", "Product of Inertia", "Center of Mass", "Wire Length", "Thermal Variance"]

for i in range(6):
    row = i // 2
    col = i % 2
    axs[row, col].plot(medianAvgCostsRL[:, i], color='tab:orange')
    axs[row, col].plot(maxAvgCostsRL[:, i], color='tab:orange', linestyle='dashed')
    axs[row, col].plot(minAvgCostsRL[:, i], color='tab:orange', linestyle='dotted')
    axs[row, col].fill_between(range(len(medianAvgCostsRL[:, i])), q1AvgCostsRL[:, i], q3AvgCostsRL[:, i], alpha=.5, linewidth=0, color='tab:orange')

    axs[row, col].plot(medianAvgCostsGA[:, i], color='tab:blue')
    axs[row, col].plot(maxAvgCostsGA[:, i], color='tab:blue', linestyle='dashed')
    axs[row, col].plot(minAvgCostsGA[:, i], color='tab:blue', linestyle='dotted')
    axs[row, col].fill_between(range(len(medianAvgCostsGA[:, i])), q1AvgCostsGA[:, i], q3AvgCostsGA[:, i], alpha=.5, linewidth=0, color='tab:blue')

    axs[row, col].plot(medianAvgCostsRS[:, i], color='tab:green')
    axs[row, col].plot(maxAvgCostsRS[:, i], color='tab:green', linestyle='dashed')
    axs[row, col].plot(minAvgCostsRS[:, i], color='tab:green', linestyle='dotted')
    axs[row, col].fill_between(range(len(medianAvgCostsRS[:, i])), q1AvgCostsRS[:, i], q3AvgCostsRS[:, i], alpha=.5, linewidth=0, color='tab:green')

    axs[row, col].set_title(cost_labels[i])
    axs[row, col].set_xlabel("Number of Function Evaluations")
    axs[row, col].set_ylabel("Average Objective")

fig.legend(["Median Average Objective RL", "Maximum Average Objective RL", "Minimum Average Objective RL", "Interquartile Average Objective RL",
            "Median Average Objective GA", "Maximum Average Objective GA", "Minimum Average Objective GA", "Interquartile Average Objective GA",
            "Median Average Objective RS", "Maximum Average Objective RS", "Minimum Average Objective RS", "Interquartile Average Objective RS"],
            loc="center right", bbox_to_anchor=(1.0, 0.5))

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(f"ResultGraphs/{date_str}/ObjectivesComparisonTransformer")

# GA Block
# HVgridGA.filterParetoFront(0,0.01)
for solutionGAIdx,solutionGA in enumerate(pfSolutionsGA):
    allDimsGA = []
    allLocsGA = []
    allTypesGA = []

    surfNormal = np.array([0, 0, 1])
    for i in range(len(componentList)):
        allTypesGA.append(componentList[i].type)

        transMat = getOrientation(int(solutionGA[4 * i + 3]))

        panelChoice = structPanelList[int(solutionGA[4 * i] % len(structPanelList))]
        if solutionGA[4 * i] >= len(structPanelList):
            surfNormal = surfNormal * -1

        surfLoc = np.matmul(panelChoice.orientation, np.multiply([solutionGA[4 * i + 1], solutionGA[4 * i + 2], surfNormal[2]], np.array(panelChoice.dimensions) / 2))
        allLocsGA.append(surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(componentList[i].dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location)

        allDimsGA.append(np.matmul(transMat, componentList[i].dimensions))

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

    objColor = tuple(np.random.rand(len(componentList), 3))

    proxyPointsGA = []
    for i in range(len(componentList)):
        xGA, yGA, zGA = getCube(allDimsGA[i], allLocsGA[i])
        ax.plot_surface(xGA, yGA, zGA, color=objColor[i], label=allTypesGA[i])
        point = ax.scatter(allLocsGA[i][0], allLocsGA[i][1], allLocsGA[i][2], color=objColor[i])
        proxyPointsGA.append(point)

    for j in range(len(structPanelList)):
        xPanel, yPanel, zPanel = getCube(panelDims[j], panelLocs[j])
        ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

    # plt.title("Visualization of Configuration GA")
    # plt.legend(proxyPointsGA, allTypesGA, loc='center left', bbox_to_anchor=(1.1, 0.5))
    if solutionGAIdx == 10:
        break
    plt.savefig(f"ResultGraphs/{date_str}/GAConfig_{solutionGAIdx}Transformer")

# RL Block
# HVgridRL.filterParetoFront(0,0.01)
for solutionRLIdx,solutionRL in enumerate(pfSolutionsRL):
    allDimsRL = []
    allLocsRL = []
    allTypesRL = []

    surfNormal = np.array([0, 0, 1])
    for i in range(len(componentList)):
        allTypesRL.append(componentList[i].type)

        transMat = getOrientation(int(solutionRL[4 * i + 3]))

        panelChoice = structPanelList[int(solutionRL[4 * i] % len(structPanelList))]
        if solutionRL[4 * i] >= len(structPanelList):
            surfNormal = surfNormal * -1

        surfLoc = np.matmul(panelChoice.orientation, np.multiply([solutionRL[4 * i + 1], solutionRL[4 * i + 2], surfNormal[2]], np.array(panelChoice.dimensions) / 2))
        allLocsRL.append(surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(componentList[i].dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location)

        allDimsRL.append(np.matmul(transMat, componentList[i].dimensions))

    panelDims = []
    panelLocs = []
    for panel in structPanelList:
        panelLocs.append(panel.location)
        panelDims.append(np.matmul(panel.orientation, panel.dimensions))

    # Create Figure for RL
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Adjustment for RL
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_aspect('equal')

    objColor = tuple(np.random.rand(len(componentList), 3))

    proxyPointsRL = []
    for i in range(len(componentList)):
        xRL, yRL, zRL = getCube(allDimsRL[i], allLocsRL[i])
        ax.plot_surface(xRL, yRL, zRL, color=objColor[i], label=allTypesRL[i])
        point = ax.scatter(allLocsRL[i][0], allLocsRL[i][1], allLocsRL[i][2], color=objColor[i])
        proxyPointsRL.append(point)

    for j in range(len(structPanelList)):
        xPanel, yPanel, zPanel = getCube(panelDims[j], panelLocs[j])
        ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

    # plt.title("Visualization of Configuration RL")
    # plt.legend(proxyPointsRL, allTypesRL, loc='center left', bbox_to_anchor=(1.1, 0.5))
    if solutionGAIdx == 10:
        break
    plt.savefig(f"ResultGraphs/{date_str}/RLConfig_{solutionRLIdx}Transformer")

# RS Block
# HVgridRS.filterParetoFront(0,0.01)
for solutionRSIdx, solutionRS in enumerate(pfSolutionsRS):
    allDimsRS = []
    allLocsRS = []
    allTypesRS = []

    surfNormal = np.array([0, 0, 1])
    for i in range(len(componentList)):
        allTypesRS.append(componentList[i].type)

        transMat = getOrientation(int(solutionRS[4 * i + 3]))

        panelChoice = structPanelList[int(solutionRS[4 * i] % len(structPanelList))]
        if solutionRS[4 * i] >= len(structPanelList):
            surfNormal = surfNormal * -1

        surfLoc = np.matmul(panelChoice.orientation, np.multiply([solutionRS[4 * i + 1], solutionRS[4 * i + 2], surfNormal[2]], np.array(panelChoice.dimensions) / 2))
        allLocsRS.append(surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(componentList[i].dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location)

        allDimsRS.append(np.matmul(transMat, componentList[i].dimensions))

    panelDims = []
    panelLocs = []
    for panel in structPanelList:
        panelLocs.append(panel.location)
        panelDims.append(np.matmul(panel.orientation, panel.dimensions))

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
        xRS, yRS, zRS = getCube(allDimsRS[i], allLocsRS[i])
        ax.plot_surface(xRS, yRS, zRS, color=objColor[i], label=allTypesRS[i])
        point = ax.scatter(allLocsRS[i][0], allLocsRS[i][1], allLocsRS[i][2], color=objColor[i])
        proxyPointsRS.append(point)

    for j in range(len(structPanelList)):
        xPanel, yPanel, zPanel = getCube(panelDims[j], panelLocs[j])
        ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

    # plt.title("Visualization of Configuration RS")
    # plt.legend(proxyPointsRS, allTypesRS, loc='center left', bbox_to_anchor=(1.1, 0.5))
    if solutionRSIdx == 10:
        break
    plt.savefig(f"ResultGraphs/{date_str}/RSConfig_{solutionRSIdx}Transformer")
