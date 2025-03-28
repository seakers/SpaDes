import numpy as np
import time
import matplotlib.pyplot as plt
from SCDesignClasses import *
from ConfigUtils import *
from ComponentList import getComponents
import datetime
import os
import pickle

print("Running LoadDataForGraphs.py")

# Choose to adjust hv/cost graphs, config plots, or both
hvcost = False
config = True

# Chose Folder to adjust graphs for
folder = "ResultGraphs/2025-03-05_14-12-35_50_Comp_150_epoch_32_batch_10_run"


componentList, transferLearningComponents = getComponents() # need to edit this to get the correct set of components

numComps = len(componentList)

graph_data = np.load(f'{folder}/graph_data.npz',allow_pickle=True)
pareto_fronts = np.load(f'{folder}/pareto_fronts.npz',allow_pickle=True)

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

print("Loaded Data")

if hvcost:

    plt.figure()
    
    plt.plot(medianHVRS, color='tab:green')
    plt.plot(maxHVRS, color='tab:green', linestyle='dashed')
    plt.plot(minHVRS, color='tab:green', linestyle='dotted')
    plt.fill_between(range(len(medianHVRS)), q1HVRS, q3HVRS, alpha=.5, linewidth=0, color='tab:green')

    plt.plot(medianHVGA, color='tab:blue')
    plt.plot(maxHVGA, color='tab:blue', linestyle='dashed')
    plt.plot(minHVGA, color='tab:blue', linestyle='dotted')
    plt.fill_between(range(len(medianHVGA)), q1HVGA, q3HVGA, alpha=.5, linewidth=0, color='tab:blue')

    plt.plot(medianHVRL, color='tab:orange')
    plt.plot(maxHVRL, color='tab:orange', linestyle='dashed')
    plt.plot(minHVRL, color='tab:orange', linestyle='dotted')
    plt.fill_between(range(len(medianHVRL)), q1HVRL, q3HVRL, alpha=.5, linewidth=0, color='tab:orange')

    plt.legend(["Median Hypervolume RS", "Maximum Hypervolume RS", "Minimum Hypervolume RS", "Interquartile Hypervolume RS",
                "Median Hypervolume GA", "Maximum Hypervolume GA", "Minimum Hypervolume GA", "Interquartile Hypervolume GA",
                "Median Hypervolume Transformer", "Maximum Hypervolume Transformer", "Minimum Hypervolume Transformer", "Interquartile Hypervolume Transformer"], 
                loc='lower right', fontsize='small')
    plt.ylim(.27, .6)
    plt.xlim(-10, 120)
    # plt.yticks(np.arange(.4, .72, 0.02))
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Comparison")
    plt.savefig(f"{folder}/HypervolumeComparison_adj")

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs[-1,-1].axis('off')
    cost_labels = ["Moment of Inertia", "Product of Inertia", "Center of Mass", "Wire Length", "Thermal Variance"]

    for i in range(5):
        row = i // 2
        col = i % 2

        axs[row, col].plot(medianAvgCostsRS[:, i], color='tab:green')
        axs[row, col].plot(maxAvgCostsRS[:, i], color='tab:green', linestyle='dashed')
        axs[row, col].plot(minAvgCostsRS[:, i], color='tab:green', linestyle='dotted')
        axs[row, col].fill_between(range(len(medianAvgCostsRS[:, i])), q1AvgCostsRS[:, i], q3AvgCostsRS[:, i], alpha=.5, linewidth=0, color='tab:green')

        axs[row, col].plot(medianAvgCostsGA[:, i], color='tab:blue')
        axs[row, col].plot(maxAvgCostsGA[:, i], color='tab:blue', linestyle='dashed')
        axs[row, col].plot(minAvgCostsGA[:, i], color='tab:blue', linestyle='dotted')
        axs[row, col].fill_between(range(len(medianAvgCostsGA[:, i])), q1AvgCostsGA[:, i], q3AvgCostsGA[:, i], alpha=.5, linewidth=0, color='tab:blue')

        axs[row, col].plot(medianAvgCostsRL[:, i], color='tab:orange')
        axs[row, col].plot(maxAvgCostsRL[:, i], color='tab:orange', linestyle='dashed')
        axs[row, col].plot(minAvgCostsRL[:, i], color='tab:orange', linestyle='dotted')
        axs[row, col].fill_between(range(len(medianAvgCostsRL[:, i])), q1AvgCostsRL[:, i], q3AvgCostsRL[:, i], alpha=.5, linewidth=0, color='tab:orange')

        axs[row, col].set_title(cost_labels[i])
        axs[row, col].set_xlabel("Episodes (Generations)")
        axs[row, col].set_ylabel("Average Objective")

    fig.legend(["Median Average Objective GA", "Maximum Average Objective GA", "Minimum Average Objective GA", "Interquartile Average Objective GA",
                "Median Average Objective RS", "Maximum Average Objective RS", "Minimum Average Objective RS", "Interquartile Average Objective RS",
                "Median Average Objective RL", "Maximum Average Objective RL", "Minimum Average Objective RL", "Interquartile Average Objective RL"],
                # loc="center right", bbox_to_anchor=(1.0, 0.5))
                loc="lower center", bbox_to_anchor=(0.6, 0))

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(f'{folder}/ObjectivesComparisonTransformer_adj')

if config:

    # GA Block
    allStructDCMs = []
    print("\nGenetic Algorithm Filtered Pareto Front")
    if len(pfSolutionsGA) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsGA) == 1:
        if len(pfSolutionsGA['Structure']) > 8:
            print("\nDesign:", pfSolutionsGA, "\nCosts:", pfPointsGA)
    else:
        for idx, solution in enumerate(pfSolutionsGA):
            if len(solution[0]['Structure']) > 8:
                print("\nDesign:", solution, "\nCosts:", pfPointsGA[idx])

    for solutionGAIdx, solDictGA in enumerate(pfSolutionsGA):
        if len(solDictGA[0]['Structure']) > 8:
            allDimsCompsGA = []
            allLocsCompsGA = []
            allOrientsCompsGA = []
            allTypesCompsGA = []
            allDimsStructGA = []
            allLocsStructGA = []
            allOrientsStructGA = []

            if isinstance(solDictGA, dict):
                structSolGA = solDictGA['Structure']
                compSolGA = solDictGA['Components']
            else:
                structSolGA = solDictGA[0]['Structure']
                compSolGA = solDictGA[0]['Components']


            structPanels = []
            numPanels = int(len(structSolGA)/8)
            for j in range(numPanels):
                panelSize = [structSolGA[8*j],structSolGA[8*j+1]]
                panelLoc = [structSolGA[8*j+2],structSolGA[8*j+3],structSolGA[8*j+4]]
                panelEAngles = [structSolGA[8*j+5],structSolGA[8*j+6],structSolGA[8*j+7]]
                panelDCM = Euler2DCM(panelEAngles)
                # print("Panel DCM: ", panelDCM)
                allStructDCMs.append(panelDCM)

                newPanel = StructPanel(dimensions=panelSize, location=panelLoc, orientation=panelDCM)
                structPanels.append(newPanel)

                allDimsStructGA.append(panelSize)
                allLocsStructGA.append(panelLoc)
                allOrientsStructGA.append(panelDCM)

            surfNormal = np.array([0,0,1])

            for i in range(numComps):
                compPanel = compSolGA[5*i]
                compLoc = [compSolGA[5*i+1],compSolGA[5*i+2]]
                compFace = compSolGA[5*i+3]
                compRot = compSolGA[5*i+4]

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
                allOrientsCompsGA.append(compDCM)

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
                allLocsCompsGA.append(compLoc)
                allDimsCompsGA.append(componentList[i].dimensions)

                # print("\nComp DCM: ", componentList[i].orientation, "\nPanelSide: ", compPanel, "\nCompFace: ", compFace, "\nRotationAngle: ", compRot)


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
                xGA, yGA, zGA = getCube(allDimsCompsGA[i], allLocsCompsGA[i], allOrientsCompsGA[i])
                # ax.plot_surface(xGA, yGA, zGA, color=objColor[i], label=allTypesCompsGA[i])
                ax.plot_surface(xGA, yGA, zGA)
                # point = ax.scatter(allLocsCompsGA[i][0], allLocsCompsGA[i][1], allLocsCompsGA[i][2], color=objColor[i])
                # proxyPointsGA.append(point)

            for j in range(len(structPanels)):
                xPanel, yPanel, zPanel = getCube(allDimsStructGA[j], allLocsStructGA[j], allOrientsStructGA[j])
                ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

            plt.title("Visualization of Configuration GA")
            # plt.legend(proxyPointsGA, allTypesGA, loc='center left', bbox_to_anchor=(1.1, 0.5))

            plt.savefig(f"{folder}/GAConfig_{solutionGAIdx}_adj")
            pickle.dump(fig, open(f"{folder}/GAConfig_{solutionGAIdx}Interactive_adj", "wb"))

    # RL Block
    allStructDCMs = []
    print("\nTransformer Filtered Pareto Front")
    if len(pfSolutionsRL) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsRL) == 1:
        if len(pfSolutionsRL['Structure']) > 8:
            print("\nDesign:", pfSolutionsRL, "\nCosts:", pfPointsRL)
    else:
        for idx, solution in enumerate(pfSolutionsRL):
            if len(solution[0]['Structure']) > 8:
                print("\nDesign:", solution, "\nCosts:", pfPointsRL[idx])
    for solutionRLIdx, solDictRL in enumerate(pfSolutionsRL):
        if len(solDictRL[0]['Structure']) > 8:
            allDimsCompsRL = []
            allLocsCompsRL = []
            allOrientsCompsRL = []
            allTypesCompsRL = []
            allDimsStructRL = []
            allLocsStructRL = []
            allOrientsStructRL = []

            if isinstance(solDictRL, dict):
                structSolRL = solDictRL['Structure']
                compSolRL = solDictRL['Components']
            else:
                structSolRL = solDictRL[0]['Structure']
                compSolRL = solDictRL[0]['Components']


            structPanels = []
            numPanels = int(len(structSolRL)/8)
            for j in range(numPanels):
                panelSize = [structSolRL[8*j],structSolRL[8*j+1]]
                panelLoc = [structSolRL[8*j+2],structSolRL[8*j+3],structSolRL[8*j+4]]
                panelEAngles = [structSolRL[8*j+5],structSolRL[8*j+6],structSolRL[8*j+7]]
                panelDCM = Euler2DCM(panelEAngles)
                # print("Panel DCM: ", panelDCM)
                allStructDCMs.append(panelDCM)

                newPanel = StructPanel(dimensions=panelSize, location=panelLoc, orientation=panelDCM)
                structPanels.append(newPanel)

                allDimsStructRL.append(panelSize)
                allLocsStructRL.append(panelLoc)
                allOrientsStructRL.append(panelDCM)

            surfNormal = np.array([0,0,1])

            for i in range(numComps):
                compPanel = compSolRL[5*i]
                compLoc = [compSolRL[5*i+1],compSolRL[5*i+2]]
                compFace = compSolRL[5*i+3]
                compRot = compSolRL[5*i+4]

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
                allOrientsCompsRL.append(compDCM)

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
                allLocsCompsRL.append(compLoc)
                allDimsCompsRL.append(componentList[i].dimensions)

                # print("\nComp DCM: ", componentList[i].orientation, "\nPanelSide: ", compPanel, "\nCompFace: ", compFace, "\nRotationAngle: ", compRot)


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
                xRL, yRL, zRL = getCube(allDimsCompsRL[i], allLocsCompsRL[i], allOrientsCompsRL[i])
                # ax.plot_surface(xRL, yRL, zRL, color=objColor[i], label=allTypesCompsRL[i])
                ax.plot_surface(xRL, yRL, zRL)
                # point = ax.scatter(allLocsCompsRL[i][0], allLocsCompsRL[i][1], allLocsCompsRL[i][2], color=objColor[i])
                # proxyPointsRL.append(point)

            for j in range(len(structPanels)):
                xPanel, yPanel, zPanel = getCube(allDimsStructRL[j], allLocsStructRL[j], allOrientsStructRL[j])
                ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

            plt.title("Visualization of Configuration RL")
            # plt.legend(proxyPointsRL, allTypesRL, loc='center left', bbox_to_anchor=(1.1, 0.5))

            plt.savefig(f"{folder}/RLConfig_{solutionRLIdx}_adj")
            pickle.dump(fig, open(f"{folder}/RLConfig_{solutionRLIdx}Interactive_adj", "wb"))


    # RS Block
    allStructDCMs = []
    print("\nRandom Search Filtered Pareto Front")
    if len(pfSolutionsRS) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsRS) == 1:
        if len(pfSolutionsRS['Structure']) > 8:
            print("\nDesign:", pfSolutionsRS, "\nCosts:", pfPointsRS)
    else:
        for idx, solution in enumerate(pfSolutionsRS):
            if len(solution[0]['Structure']) > 8:
                print("\nDesign:", solution, "\nCosts:", pfPointsRS[idx])

    for solutionRSIdx, solDictRS in enumerate(pfSolutionsRS):
        if len(solDictRS[0]['Structure']) > 8:
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
                allStructDCMs.append(panelDCM)

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

            plt.savefig(f"{folder}/RSConfig_{solutionRSIdx}_adj")
            pickle.dump(fig, open(f"{folder}/RSConfig_{solutionRSIdx}Interactive_adj", "wb"))
