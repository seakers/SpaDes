import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ConfigurationOptimization import *
from SCDesignClasses import Component
from ConfigurationCost import maxCostComps

def main():
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

    # componentList = [
    #     Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5),
    #     Component(type="transmitter", mass=4.5, dimensions=[.25,.15,.05], heatDisp=12),
    #     Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10),
    #     Component(type="reciever", mass=4, dimensions=[.2,.15,.03], heatDisp=11),
    #     Component(type="reciever", mass=3, dimensions=[.175,.125,.03], heatDisp=9),
    #     Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7),
    #     Component(type="OBDH", mass=9, dimensions=[.24,.18,.18], heatDisp=6),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5),
    #     Component(type="payload", mass=5, dimensions=[.3,.25,.2], heatDisp=3),
    #     Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5),
    #     Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5)
    # ]

    # componentList = [
    #     Component(type="fuel tank", mass=15, dimensions=[.3,.25,.2], heatDisp=3),
    #     Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #     Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #     Component(type="star tracker", mass=1.8, dimensions=[.1,.15,.12], heatDisp=1),
    #     Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.2),
    #     Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.2),
    #     Component(type="antenna", mass=3.5, dimensions=[.3,.12,.1], heatDisp=8),
    #     Component(type="antenna", mass=3, dimensions=[.25,.1,.08], heatDisp=7),
    #     Component(type="data recorder", mass=2.5, dimensions=[.22,.18,.06], heatDisp=6),
    #     Component(type="PCU", mass=6.5, dimensions=[.25,.2,.12], heatDisp=6),
    #     Component(type="OBDH", mass=8, dimensions=[.22,.18,.16], heatDisp=5),
    #     Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #     Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #     Component(type="radiation sensor", mass=1.2, dimensions=[.1,.08,.03], heatDisp=1.3),
    #     Component(type="payload", mass=4.5, dimensions=[.28,.22,.18], heatDisp=2.8),
    #     Component(type="solar panel", mass=1.2, dimensions=[.18,.48,.01], heatDisp=1.4),
    #     Component(type="solar panel", mass=1.2, dimensions=[.18,.48,.01], heatDisp=1.4),
    #     Component(type="battery", mass=6, dimensions=[.2,.18,.12], heatDisp=2)
    # ]

    # componentList = [
    #     Component(type="OBDH", mass=3, dimensions=[.24,.19,.07], heatDisp=7),
    #     Component(type="PCU", mass=7, dimensions=[.26,.21,.13], heatDisp=6.5),
    #     Component(type="command processor", mass=7.5, dimensions=[.23,.19,.17], heatDisp=5.5),
    #     Component(type="inertial measurement unit", mass=2.3, dimensions=[.14,.11,.09], heatDisp=2.5),
    #     Component(type="inertial measurement unit", mass=2.3, dimensions=[.14,.11,.09], heatDisp=2.5),
    #     Component(type="antenna", mass=4.2, dimensions=[.35,.14,.12], heatDisp=9),
    #     Component(type="antenna", mass=3.2, dimensions=[.22,.09,.07], heatDisp=7.5),
    #     Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #     Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #     Component(type="gamma-ray sensor", mass=1.3, dimensions=[.11,.09,.04], heatDisp=1.4),
    #     Component(type="payload", mass=5, dimensions=[.3,.24,.2], heatDisp=3),
    #     Component(type="solar panel", mass=1.3, dimensions=[.19,.49,.01], heatDisp=1.5),
    #     Component(type="solar panel", mass=1.3, dimensions=[.19,.49,.01], heatDisp=1.5),
    #     Component(type="battery", mass=5.5, dimensions=[.22,.2,.13], heatDisp=2.2),
    #     Component(type="propellant tank", mass=12, dimensions=[.28,.22,.18], heatDisp=4),
    #     Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9),
    #     Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9),
    #     Component(type="sun sensor", mass=1.5, dimensions=[.09,.14,.1], heatDisp=0.9)
    # ]

    # componentList = [
    #     Component(type="PCU", mass=6, dimensions=[.25,.2,.12], heatDisp=6.2),
    #     Component(type="OBDH", mass=8.5, dimensions=[.24,.19,.16], heatDisp=5.8),
    #     Component(type="flight computer", mass=7.8, dimensions=[.23,.19,.18], heatDisp=5.6),
    #     Component(type="reaction wheel", mass=2.8, dimensions=[.14,.12,.1], heatDisp=3),
    #     Component(type="reaction wheel", mass=2.8, dimensions=[.14,.12,.1], heatDisp=3),
    #     Component(type="antenna", mass=4.5, dimensions=[.34,.15,.12], heatDisp=9.5),
    #     Component(type="antenna", mass=3.3, dimensions=[.23,.1,.08], heatDisp=7.8),
    #     Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #     Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #     Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.09], heatDisp=1.1),
    #     Component(type="payload", mass=5.2, dimensions=[.29,.24,.21], heatDisp=3.2),
    #     Component(type="solar panel", mass=1.4, dimensions=[.18,.47,.01], heatDisp=1.6),
    #     Component(type="solar panel", mass=1.4, dimensions=[.18,.47,.01], heatDisp=1.6),
    #     Component(type="battery", mass=5.8, dimensions=[.22,.21,.14], heatDisp=2.3),
    #     Component(type="hydrazine tank", mass=13, dimensions=[.29,.23,.19], heatDisp=4.1),
    #     Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1),
    #     Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1),
    #     Component(type="magnetometer", mass=1.4, dimensions=[.09,.13,.1], heatDisp=1)
    # ]

    componentList = [
        Component(type="reaction wheel", mass=3, dimensions=[.14,.13,.1], heatDisp=3.1),
        Component(type="reaction wheel", mass=3, dimensions=[.14,.13,.1], heatDisp=3.1),
        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.2),
        Component(type="antenna", mass=4.3, dimensions=[.33,.15,.12], heatDisp=9.7),
        Component(type="antenna", mass=3.4, dimensions=[.22,.09,.08], heatDisp=7.9),
        Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
        Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
        Component(type="magnetometer", mass=1.5, dimensions=[.1,.12,.1], heatDisp=1.1),
        Component(type="solar panel", mass=1.5, dimensions=[.19,.48,.01], heatDisp=1.7),
        Component(type="solar panel", mass=1.5, dimensions=[.19,.48,.01], heatDisp=1.7),
        Component(type="battery", mass=6, dimensions=[.23,.2,.14], heatDisp=2.5),
        Component(type="propellant tank", mass=12.5, dimensions=[.28,.22,.18], heatDisp=4.2),
        Component(type="PCU", mass=6.3, dimensions=[.26,.21,.13], heatDisp=6.4),
        Component(type="OBDH", mass=8.2, dimensions=[.24,.18,.15], heatDisp=5.9),
        Component(type="flight computer", mass=7.6, dimensions=[.22,.19,.17], heatDisp=5.7),
        Component(type="payload", mass=5.5, dimensions=[.3,.23,.22], heatDisp=3.4)
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
    t00 = time.time()
    allHVGA = []
    allAvgCostsGA = []
    for runGA in range(numRuns):
        print("\n\n\nRUN: ", runGA, "\n\n")
        numStepsGA, allHVGARun, HVgridGA, avgCostsGA = optimization(componentList,structPanelList,maxCostList,"GA")
        allHVGA.append(allHVGARun)
        allAvgCostsGA.append(avgCostsGA)
    t01 = time.time()

    # Reset locations and dimensions for Reinforcement Learning
    i = 0
    for comp in componentList:
        comp.location = compLocs[i]
        comp.dimensions = compDims[i]
        i+=1
    t10 = time.time()
    allHVRL = []
    allAvgCostsRL = []
    for runRL in range(numRuns):
        print("RUN: ", runRL, "\n\n")
        numStepsRL, allHVRLRun, HVgridRL, avgCostsRL = optimization(componentList,structPanelList,maxCostList,"RL")
        allHVRL.append(allHVRLRun)
        allAvgCostsRL.append(avgCostsRL)
    t11 = time.time()

    print("GA Average Time: ", (t01-t00)/numRuns)
    print("RL Average Time: ", (t11-t10)/numRuns)

    allHVGA = np.array(allHVGA)
    allHVRL = np.array(allHVRL)
    allAvgCostsGA = np.array(allAvgCostsGA)
    allAvgCostsRL = np.array(allAvgCostsRL)

    medianHVGA = np.median(allHVGA,0)
    medianHVRL = np.median(allHVRL,0)
    q1HVGA = np.quantile(allHVGA,.25,axis=0)
    q3HVGA = np.quantile(allHVGA,.75,axis=0)
    q1HVRL = np.quantile(allHVRL,.25,axis=0)
    q3HVRL = np.quantile(allHVRL,.75,axis=0)
    maxHVGA = np.max(allHVGA,0)
    minHVGA = np.min(allHVGA,0)
    maxHVRL = np.max(allHVRL,0)
    minHVRL = np.min(allHVRL,0)

    medianAvgCostsGA = np.median(allAvgCostsGA,0)
    medianAvgCostsRL = np.median(allAvgCostsRL,0)
    q1AvgCostsGA = np.quantile(allAvgCostsGA,.25,axis=0)
    q3AvgCostsGA = np.quantile(allAvgCostsGA,.75,axis=0)
    q1AvgCostsRL = np.quantile(allAvgCostsRL,.25,axis=0)
    q3AvgCostsRL = np.quantile(allAvgCostsRL,.75,axis=0)
    maxAvgCostsGA = np.max(allAvgCostsGA,0)
    minAvgCostsGA = np.min(allAvgCostsGA,0)
    maxAvgCostsRL = np.max(allAvgCostsRL,0)
    minAvgCostsRL = np.min(allAvgCostsRL,0)

    plt.figure()
    plt.plot(medianHVGA,color='tab:blue')
    plt.plot(medianHVRL,color='tab:orange')
    plt.plot(maxHVGA,color='tab:blue',linestyle='dashed')
    plt.plot(minHVGA,color='tab:blue',linestyle='dotted')
    plt.plot(maxHVRL,color='tab:orange',linestyle='dashed')
    plt.plot(minHVRL,color='tab:orange',linestyle='dotted')
    plt.fill_between(range(len(medianHVGA)), q1HVGA, q3HVGA, alpha=.5, linewidth=0, color='tab:blue')
    plt.fill_between(range(len(medianHVRL)), q1HVRL, q3HVRL, alpha=.5, linewidth=0, color='tab:orange')
    plt.legend(["Median Hypervolume GA","Median Hypervolume RL","Maximum Hypervolume GA","Minimum Hypervolume GA",
                "Maximum Hypervolume RL","Minimum Hypervolume RL", "Interquartile Hypervolume GA","Interquartile Hypervolume RL"],loc="lower right")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.title("Deep RL / Genetic Algorithm Hypervolume Comparison")
    plt.show()

    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # Increase the figure width to make space for the legend
    cost_labels = ["Overlap", "Moment of Inertia", "Product of Inertia", "Center of Mass", "Wire Length", "Thermal Variance"]

    for i in range(6):
        row = i // 2
        col = i % 2
        axs[row, col].plot(medianAvgCostsGA[:, i], color='tab:blue')
        axs[row, col].plot(medianAvgCostsRL[:, i], color='tab:orange')
        axs[row, col].plot(maxAvgCostsGA[:, i], color='tab:blue', linestyle='dashed')
        axs[row, col].plot(minAvgCostsGA[:, i], color='tab:blue', linestyle='dotted')
        axs[row, col].plot(maxAvgCostsRL[:, i], color='tab:orange', linestyle='dashed')
        axs[row, col].plot(minAvgCostsRL[:, i], color='tab:orange', linestyle='dotted')
        axs[row, col].fill_between(range(len(medianAvgCostsGA[:, i])), q1AvgCostsGA[:, i], q3AvgCostsGA[:, i], alpha=.5, linewidth=0, color='tab:blue')
        axs[row, col].fill_between(range(len(medianAvgCostsRL[:, i])), q1AvgCostsRL[:, i], q3AvgCostsRL[:, i], alpha=.5, linewidth=0, color='tab:orange')
        axs[row, col].set_title(cost_labels[i])
        axs[row, col].set_xlabel("Number of Function Evaluations")
        axs[row, col].set_ylabel("Average Objective")

    # Move the legend closer to the plot
    fig.legend(["Median Average Objective GA", "Median Average Objective RL", 
                "Maximum Average Objective GA", "Minimum Average Objective GA",
                "Maximum Average Objective RL", "Minimum Average Objective RL", 
                "Interquartile Average Objective GA", "Interquartile Average Objective RL"], 
            loc="center right", bbox_to_anchor=(1.0, 0.5))

    # Adjust the layout to accommodate the legend
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Reduce the plot area to leave space for the legend
    plt.show()
        

    # GA Block
    # HVgridGA.filterParetoFront(0,0.01)
    
    pfSolutionsGA = HVgridGA.paretoFrontSolution
    pfPointsGA = HVgridGA.paretoFrontPoint
    print("\nGenetic Algorithm Filtered Pareto Front")
    for idx,solution in enumerate(pfSolutionsGA):
        print("\nDesign:", solution, "\nCosts:", pfPointsGA[idx])
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

        plt.title("Visualization of Configuration GA")
        plt.legend(proxyPointsGA, allTypesGA, loc='center left', bbox_to_anchor=(1, 0.5))
        if solutionGAIdx == 10:
            break
    plt.show()

    # RL Block
    # HVgridRL.filterParetoFront(0,0.01)
    pfSolutionsRL = HVgridRL.paretoFrontSolution
    pfPointsRL = HVgridRL.paretoFrontPoint
    print("\nDeep Reinforcement Learning Filtered Pareto Front")
    for idx,solution in enumerate(pfSolutionsRL):
        print("\nDesign:", solution, "\nCosts:", pfPointsRL[idx])
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

        plt.title("Visualization of Configuration RL")
        plt.legend(proxyPointsRL, allTypesRL, loc='center left', bbox_to_anchor=(1, 0.5))
        if solutionGAIdx == 10:
            break
    plt.show()


if __name__ == "__main__":
    main()