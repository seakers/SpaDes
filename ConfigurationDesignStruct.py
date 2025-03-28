import numpy as np
import time
from ConfigurationOptimizationStruct import * 
from RLOptTransformerStruct import trainsformerRLTraining
from SCDesignClasses import Component
from ConfigUtils import *
from ComponentList import getComponents
import datetime
import os
import pickle
import matplotlib.pyplot as plt


def main():
    print("Started Actual Code!")

    componentList, transferLearningComponents = getComponents()

    # electrical ports are located on the -x side and pointing is located on the +x side, as the dimensions are defined

    # params
    minibatch = 32 # max 32 because increasing memory apparently does nothing :(
    epochs = 150
    clipping = 0.2
    KL = 0.003
    gamma = 0.999
    lam = 0.95
    lr = 0.01
    iterations = 5
    params = [minibatch,epochs,clipping,KL,gamma,lam,lr,iterations]

    numComps = len(componentList)
    compLocs = np.ndarray.tolist(np.random.normal(0,0.4,(numComps,3)))
    compDims = []
    i = 0
    for comp in componentList:
        comp.location = compLocs[i]
        comp.orientation = np.eye(3)
        compDims.append(comp.dimensions)
        i+=1

    # Optimize
    numRuns = 1

    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'ResultGraphs/{date_str}', exist_ok=True)

    # Random Search
    t00 = time.time()
    allHVRS = []
    allAvgCostsRS = []
    for runRS in range(numRuns):
        print("\nRUN: ", runRS, "\n")
        numStepsRS, allHVRSRun, HVgridRS, avgCostsRS, maxCosts = randomSearch(componentList,params)
        allHVRS.append(allHVRSRun)
        allAvgCostsRS.append(avgCostsRS)
    t01 = time.time()

    # Reset locations and dimensions for Genetic Algorithm
    i = 0
    for comp in componentList:
        comp.location = compLocs[i]
        comp.dimensions = compDims[i]
        i+=1

    # Genetic Algorithm
    t10 = time.time()
    allHVGA = []
    allAvgCostsGA = []
    for runGA in range(numRuns):
        print("\n\n\nRUN: ", runGA, "\n\n")
        numStepsGA, allHVGARun, HVgridGA, avgCostsGA = GAOptimization(componentList,maxCosts,params)
        allHVGA.append(allHVGARun)
        allAvgCostsGA.append(avgCostsGA)
    t11 = time.time()

    # Reset locations and dimensions for Reinforcement Learning
    i = 0
    for comp in componentList:
        comp.location = compLocs[i]
        comp.dimensions = compDims[i]
        i+=1
    t20 = time.time()
    allHVRL = []
    allAvgCostsRL = []
    for runRL in range(numRuns):
        print("RUN: ", runRL, "\n\n")
        numStepsRL, allHVRLRun, HVgridRL, avgCostsRL = trainsformerRLTraining(componentList,maxCosts,date_str,params)
        allHVRL.append(allHVRLRun)
        allAvgCostsRL.append(avgCostsRL)
    t21 = time.time()

    print("RS Average Time: ", (t01-t00)/numRuns)
    print("GA Average Time: ", (t11-t10)/numRuns)
    print("RL Average Time: ", (t21-t20)/numRuns)

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

    ### Random Search Processsing

    allHVRS = np.array(allHVRS)
    allAvgCostsRS = np.array(allAvgCostsRS)

    medianHVRS = np.median(allHVRS, 0)
    q1HVRS = np.quantile(allHVRS, .25, axis=0)
    q3HVRS = np.quantile(allHVRS, .75, axis=0)
    maxHVRS = np.max(allHVRS, 0)
    minHVRS = np.min(allHVRS, 0)

    # print(allAvgCostsRS.shape)
    # print(allAvgCostsRS)
    medianAvgCostsRS = np.median(allAvgCostsRS, 0)
    q1AvgCostsRS = np.quantile(allAvgCostsRS, .25, axis=0)
    q3AvgCostsRS = np.quantile(allAvgCostsRS, .75, axis=0)
    maxAvgCostsRS = np.max(allAvgCostsRS, 0)
    minAvgCostsRS = np.min(allAvgCostsRS, 0)

    plt.figure()
    plt.plot(medianHVGA, color='tab:blue')
    plt.plot(medianHVRL, color='tab:orange')
    plt.plot(medianHVRS, color='tab:green')
    plt.plot(maxHVGA, color='tab:blue', linestyle='dashed')
    plt.plot(minHVGA, color='tab:blue', linestyle='dotted')
    plt.plot(maxHVRL, color='tab:orange', linestyle='dashed')
    plt.plot(minHVRL, color='tab:orange', linestyle='dotted')
    plt.plot(maxHVRS, color='tab:green', linestyle='dashed')
    plt.plot(minHVRS, color='tab:green', linestyle='dotted')
    plt.fill_between(range(len(medianHVGA)), q1HVGA, q3HVGA, alpha=.5, linewidth=0, color='tab:blue')
    plt.fill_between(range(len(medianHVRL)), q1HVRL, q3HVRL, alpha=.5, linewidth=0, color='tab:orange')
    plt.fill_between(range(len(medianHVRS)), q1HVRS, q3HVRS, alpha=.5, linewidth=0, color='tab:green')
    plt.legend(["Median Hypervolume GA", "Median Hypervolume RL", "Median Hypervolume RS", "Maximum Hypervolume GA", "Minimum Hypervolume GA",
                "Maximum Hypervolume RL", "Minimum Hypervolume RL", "Maximum Hypervolume RS", "Minimum Hypervolume RS",
                "Interquartile Hypervolume GA", "Interquartile Hypervolume RL", "Interquartile Hypervolume RS"], 
                loc='lower right', fontsize='small')
    # plt.ylim(.4, .7)
    # plt.yticks(np.arange(.4, .72, 0.02))
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.title("Deep RL / Genetic Algorithm / Random Search Hypervolume Comparison")
    plt.savefig(f"ResultGraphs/{date_str}/HypervolumeComparison")
    # Save data needed to recreate the graph
    np.savez(f"ResultGraphs/{date_str}/graph_data.npz",
             allHVGA=allHVGA,
             allHVRL=allHVRL,
             allHVRS=allHVRS,
             allAvgCostsGA=allAvgCostsGA,
             allAvgCostsRL=allAvgCostsRL,
             allAvgCostsRS=allAvgCostsRS,
             medianHVGA=medianHVGA,
             medianHVRL=medianHVRL,
             medianHVRS=medianHVRS,
             q1HVGA=q1HVGA,
             q3HVGA=q3HVGA,
             q1HVRL=q1HVRL,
             q3HVRL=q3HVRL,
             q1HVRS=q1HVRS,
             q3HVRS=q3HVRS,
             maxHVGA=maxHVGA,
             minHVGA=minHVGA,
             maxHVRL=maxHVRL,
             minHVRL=minHVRL,
             maxHVRS=maxHVRS,
             minHVRS=minHVRS,
             medianAvgCostsGA=medianAvgCostsGA,
             medianAvgCostsRL=medianAvgCostsRL,
             medianAvgCostsRS=medianAvgCostsRS,
             q1AvgCostsGA=q1AvgCostsGA,
             q3AvgCostsGA=q3AvgCostsGA,
             q1AvgCostsRL=q1AvgCostsRL,
             q3AvgCostsRL=q3AvgCostsRL,
             q1AvgCostsRS=q1AvgCostsRS,
             q3AvgCostsRS=q3AvgCostsRS,
             maxAvgCostsGA=maxAvgCostsGA,
             minAvgCostsGA=minAvgCostsGA,
             maxAvgCostsRL=maxAvgCostsRL,
             minAvgCostsRL=minAvgCostsRL,
             maxAvgCostsRS=maxAvgCostsRS,
             minAvgCostsRS=minAvgCostsRS)

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    # cost_labels = ["Overlap", "Moment of Inertia", "Product of Inertia", "Center of Mass", "Wire Length", "Thermal Variance"]
    cost_labels = ["Moment of Inertia", "Product of Inertia", "Center of Mass", "Wire Length", "Thermal Variance"]


    for i in range(5):
        row = i // 2
        col = i % 2
        axs[row, col].plot(medianAvgCostsGA[:, i], color='tab:blue')
        axs[row, col].plot(medianAvgCostsRL[:, i], color='tab:orange')
        axs[row, col].plot(medianAvgCostsRS[:, i], color='tab:green')
        axs[row, col].plot(maxAvgCostsGA[:, i], color='tab:blue', linestyle='dashed')
        axs[row, col].plot(minAvgCostsGA[:, i], color='tab:blue', linestyle='dotted')
        axs[row, col].plot(maxAvgCostsRL[:, i], color='tab:orange', linestyle='dashed')
        axs[row, col].plot(minAvgCostsRL[:, i], color='tab:orange', linestyle='dotted')
        axs[row, col].plot(maxAvgCostsRS[:, i], color='tab:green', linestyle='dashed')
        axs[row, col].plot(minAvgCostsRS[:, i], color='tab:green', linestyle='dotted')
        axs[row, col].fill_between(range(len(medianAvgCostsGA[:, i])), q1AvgCostsGA[:, i], q3AvgCostsGA[:, i], alpha=.5, linewidth=0, color='tab:blue')
        axs[row, col].fill_between(range(len(medianAvgCostsRL[:, i])), q1AvgCostsRL[:, i], q3AvgCostsRL[:, i], alpha=.5, linewidth=0, color='tab:orange')
        axs[row, col].fill_between(range(len(medianAvgCostsRS[:, i])), q1AvgCostsRS[:, i], q3AvgCostsRS[:, i], alpha=.5, linewidth=0, color='tab:green')
        axs[row, col].set_title(cost_labels[i])
        axs[row, col].set_xlabel("Episodes/generations")
        axs[row, col].set_ylabel("Average Objective")

    fig.legend(["Median Average Objective GA", "Median Average Objective RL", "Median Average Objective RS",
                "Maximum Average Objective GA", "Minimum Average Objective GA",
                "Maximum Average Objective RL", "Minimum Average Objective RL",
                "Maximum Average Objective RS", "Minimum Average Objective RS",
                "Interquartile Average Objective GA", "Interquartile Average Objective RL", "Interquartile Average Objective RS"],
                loc="center right", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(f"ResultGraphs/{date_str}/ObjectivesComparison")
        ########################################################################################################################################

    # GA Block
    pfSolutionsGA = HVgridGA.paretoFrontSolution
    pfPointsGA = HVgridGA.paretoFrontPoint
    allStructDCMs = []
    print("\nGenetic Algorithm Filtered Pareto Front")
    if len(pfSolutionsGA) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsGA) == 1:
        print("\nDesign:", pfSolutionsGA, "\nCosts:", pfPointsGA)
    else:
        for idx, solution in enumerate(pfSolutionsGA):
            print("\nDesign:", solution, "\nCosts:", pfPointsGA[idx])
            if idx == 5:
                break
    for solutionGAIdx, solDictGA in enumerate(pfSolutionsGA):
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

        plt.savefig(f"ResultGraphs/{date_str}/GAConfig_{solutionGAIdx}")
        pickle.dump(fig, open(f"ResultGraphs/{date_str}/GAConfig_{solutionGAIdx}Interactive", "wb"))
        if solutionGAIdx == 5:
            break

    # RL Block
    pfSolutionsRL = HVgridRL.paretoFrontSolution
    pfPointsRL = HVgridRL.paretoFrontPoint
    allStructDCMs = []
    print("\nTransformer Filtered Pareto Front")
    if len(pfSolutionsRL) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsRL) == 1:
        print("\nDesign:", pfSolutionsRL, "\nCosts:", pfPointsRL)
    else:
        for idx, solution in enumerate(pfSolutionsRL):
            print("\nDesign:", solution, "\nCosts:", pfPointsRL[idx])
            if idx == 5:
                break
    for solutionRLIdx, solDictRL in enumerate(pfSolutionsRL):
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

        plt.savefig(f"ResultGraphs/{date_str}/RLConfig_{solutionRLIdx}")
        pickle.dump(fig, open(f"ResultGraphs/{date_str}/RLConfig_{solutionRLIdx}Interactive", "wb"))
        if solutionRLIdx == 5:
            break


    # RS Block
    pfSolutionsRS = HVgridRS.paretoFrontSolution
    pfPointsRS = HVgridRS.paretoFrontPoint
    allStructDCMs = []
    print("\nRandom Search Filtered Pareto Front")
    if len(pfSolutionsRS) == 0:
        print("\nNo Valid Designs\n")
    elif len(pfSolutionsRS) == 1:
        print("\nDesign:", pfSolutionsRS, "\nCosts:", pfPointsRS)
    else:
        for idx, solution in enumerate(pfSolutionsRS):
            print("\nDesign:", solution, "\nCosts:", pfPointsRS[idx])
            if idx == 5:
                break
    for solutionRSIdx, solDictRS in enumerate(pfSolutionsRS):
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

        plt.savefig(f"ResultGraphs/{date_str}/RSConfig_{solutionRSIdx}")
        pickle.dump(fig, open(f"ResultGraphs/{date_str}/RSConfig_{solutionRSIdx}Interactive", "wb"))
        if solutionRSIdx == 5:
            break

    # Save Pareto fronts for each method
    np.savez(f"ResultGraphs/{date_str}/pareto_fronts.npz",
             pfSolutionsGA=pfSolutionsGA,
             pfPointsGA=pfPointsGA,
             pfSolutionsRL=pfSolutionsRL,
             pfPointsRL=pfPointsRL,
             pfSolutionsRS=pfSolutionsRS,
             pfPointsRS=pfPointsRS)

    


if __name__ == "__main__":
    main()