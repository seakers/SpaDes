import numpy as np
import pygad
from copy import deepcopy
from ConfigurationCostStruct import getCostCompsNonnormal, getCostComps
from ConfigUtils import *
from HypervolumeUtils import HypervolumeGrid
from SCDesignClasses import Component, StructPanel

def GAFitnessFunc(GAInstance,solution,solutionIDX):
    # Fitness function for pyGAD. Calls the cost function and inverts the output
    components = deepcopy(compList)
    numComps = len(components)

    # adjust solution to fit needed parameters
    numPanels = int(solution[0])

    structSol = np.zeros(numPanels*8)
    structSol[:8] = [1,1,0,0,-1,0,0,0] # lv adapter panel

    structMult = np.tile(np.array([2,2,2,2,2,2*np.pi,2*np.pi,2*np.pi]),(numPanels-1))
    structAdd = np.tile(np.array([0,0,-1,-1,-1,0,0,0]),(numPanels-1))
    structSol[8:] = np.array(solution[1:(numPanels-1)*8+1])*structMult + structAdd

    compMult = np.tile(np.array([1,2,2,1,2*np.pi]),numComps)
    compAdd = np.tile(np.array([0,-1,-1,0,0]),numComps)
    compSol = np.array(solution[-5*numComps:])*compMult + compAdd

    # Change the continuous panel choice variable to a discrete variable by binning
    panelChoices = 2*numPanels-1
    bins = (np.arange(panelChoices) + 1)/panelChoices
    compSol[-5*numComps::5] = np.digitize(compSol[-5*numComps::5],bins,right=True)

    solutionDict = {"Structure":structSol, "Components":compSol}

    structPanels = []
    numPanels = int(len(structSol)/8)
    for j in range(numPanels):
        panelSize = [structSol[8*j],structSol[8*j+1]]
        panelLoc = [structSol[8*j+2],structSol[8*j+3],structSol[8*j+4]]
        panelEAngles = [structSol[8*j+5],structSol[8*j+6],structSol[8*j+7]]
        panelDCM = Euler2DCM(panelEAngles)

        newPanel = StructPanel(dimensions=panelSize, location=panelLoc, orientation=panelDCM)
        structPanels.append(newPanel)

    surfNormal = np.array([0,0,1])

    for i in range(len(components)):
        compPanel = compSol[5*i]
        compLoc = [compSol[5*i+1],compSol[5*i+2]]
        compFace = compSol[5*i+3]
        compRot = compSol[5*i+4]

        compOrient = faceAndRot2DCM(faceChoice=compFace, rot=compRot)

        if compPanel >= len(structPanels): # plus one because the first choice is the top of the lv adapter, and no comps can go on the bottom
            panelChoice = structPanels[int(compPanel - len(structPanels)) + 1]
            panelChoiceDCM = panelChoice.orientation
            panelChoiceDCM = np.matmul(panelChoiceDCM,np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
            surfNormal = surfNormal*-1
        else:
            panelChoice = structPanels[int(compPanel)]
            panelChoiceDCM = panelChoice.orientation
        
        compDCM = align_cuboid_with_plane(compOrient, panelChoiceDCM)

        components[i].orientation = compDCM
        
        if compFace%3 == 0:
            dimOffset = components[i].dimensions[2]/2
        elif compFace%3 == 1:
            dimOffset = components[i].dimensions[1]/2
        elif compFace%3 == 2:
            dimOffset = components[i].dimensions[0]/2

        panelOffset = panelChoice.thickness/2
        offsetVect = np.matmul(panelChoiceDCM,surfNormal*(dimOffset+panelOffset))
        
        surfLoc = np.matmul(panelChoiceDCM,np.multiply([compLoc[0],compLoc[1],surfNormal[2]],np.array(panelChoice.dimensions)/2))
        compLoc = surfLoc + offsetVect + panelChoice.location
        components[i].location = compLoc

            
    costList, constraint = getCostComps(components,structPanels, maxCosts)
    rewardList = -np.array(costList)

    global NFE
    global allCosts
    global HVgrid
    global allHV
    NFE+=1
    allCosts.append(rewardList)
    if not constraint:
        HVgrid.updateHV(costList,solutionDict)
    allHV.append(HVgrid.getHV())

    returnRewards = np.zeros(6)
    returnRewards[1:] = rewardList
    if constraint: # constraint handling to prevent overlap
        returnRewards[0] = -100
    
    return returnRewards

def on_generation(ga_instance):
    global last_fitness
    global allCosts
    global avgCosts
    genAvgCost = np.mean(np.array(allCosts),0)
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Avg Cost = {genAvgCost}\n")
    avgCosts.append(genAvgCost)
    allCosts = []

def GAOptimization(componentList,maxCostList,params):
    # uses a GA to find the optimal spacecraft configuration
    # Parameters
    num_generations = params[1] # Number of generations, is set equal to num epochs
    sol_per_pop = params[0] # Number of solutions in the population. Is set equal to batch size

    num_parents_mating = int(sol_per_pop/4) # Number of solutions to be selected as parents in the mating pool.

    maxPanels = 10
    numComponents = len(componentList)

    num_genes = 1 + (maxPanels-1)*8 + numComponents*5 # 1 choice of how many panels + 9 possible panels(10 - 1 preset one) * 8 decisions per panel + 
                                            # number of components + 5 decisions per component
    save_best_solutions = True
    gene_space = [{'low': 0, 'high': 1}] * num_genes

    gene_space[0] = np.arange(maxPanels)+1 # choose number of panels. Plus one to make it go from 1 to 10 instead of 0 to 9
    for comp in range(numComponents):
        gene_space[-5*comp - 2] = np.arange(6) # choose face on panel. One choice for each side of a cuboid

    # all other decisions are continuous so they are left as "None". 
    # Continuous ranges will be transformed later from [0,1] to the appropriate range
    # Panel choice for components is not continuous but it will be decided by binning the continuous output
    # this is done because there can be a variable number of panels

    parent_selection_type = "nsga2"

    global compList
    compList = componentList
    global maxCosts
    maxCosts = maxCostList
    global last_fitness
    last_fitness = 0

    global NFE
    global allCosts
    global avgCosts
    global HVgrid
    global allHV

    NFE = 0
    allHV = []
    allCosts = []
    avgCosts = []
    HVgrid = HypervolumeGrid([1,1,1,1,1]) # Only 5 to eliminate constraint (overlap cost) from HV calculation
    allHV = []

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       fitness_func=GAFitnessFunc,
                       on_generation=on_generation)
    
    # Running the GA to optimize the parameters of the function.
    ga_instance.run()
        
    return num_generations,allHV,HVgrid,avgCosts

def randSearchCostCalc(components,solution):
    structSol = solution["Structure"]
    compSol = solution["Components"]

    structPanels = []
    numPanels = int(len(structSol)/8)
    for j in range(numPanels):
        panelSize = [structSol[8*j],structSol[8*j+1]]
        panelLoc = [structSol[8*j+2],structSol[8*j+3],structSol[8*j+4]]
        panelEAngles = [structSol[8*j+5],structSol[8*j+6],structSol[8*j+7]]
        panelDCM = Euler2DCM(panelEAngles)

        newPanel = StructPanel(dimensions=panelSize, location=panelLoc, orientation=panelDCM)
        structPanels.append(newPanel)

    surfNormal = np.array([0,0,1])

    for i in range(len(components)):
        compPanel = compSol[5*i]
        compLoc = [compSol[5*i+1],compSol[5*i+2]]
        compFace = compSol[5*i+3]
        compRot = compSol[5*i+4]

        compOrient = faceAndRot2DCM(faceChoice=compFace, rot=compRot)

        if compPanel >= len(structPanels): # plus one because the first choice is the top of the lv adapter, and no comps can go on the bottom
            panelChoice = structPanels[int(compPanel - len(structPanels)) + 1]
            panelChoiceDCM = panelChoice.orientation
            panelChoiceDCM = np.matmul(panelChoiceDCM,np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
            surfNormal = surfNormal*-1
        else:
            panelChoice = structPanels[int(compPanel)]
            panelChoiceDCM = panelChoice.orientation
        
        compDCM = align_cuboid_with_plane(compOrient, panelChoiceDCM)

        components[i].orientation = compDCM
        
        if compFace%3 == 0:
            dimOffset = components[i].dimensions[2]/2
        elif compFace%3 == 1:
            dimOffset = components[i].dimensions[1]/2
        elif compFace%3 == 2:
            dimOffset = components[i].dimensions[0]/2

        panelOffset = panelChoice.thickness/2
        offsetVect = np.matmul(panelChoiceDCM,surfNormal*(dimOffset+panelOffset))
        
        surfLoc = np.matmul(panelChoiceDCM,np.multiply([compLoc[0],compLoc[1],surfNormal[2]],np.array(panelChoice.dimensions)/2))
        compLoc = surfLoc + offsetVect + panelChoice.location
        components[i].location = compLoc

            
    costList, constraint = getCostCompsNonnormal(components,structPanels)
    return costList, constraint

def randomSearch(components, params):
    # Random Search to find max costs. will multiply those by 1.1 and treat as reference point for hv calcs.
    numRuns = params[0]*params[1] # batch size * epochs
    numBatches = params[0]
    compLength = len(components)

    allCosts = []
    allConstraints = []
    allSolutions = []
    allHV = []

    for i in range(numRuns):
        structSol = []
        compSol = []
        numPanels = np.random.choice(np.arange(10)+1) # Choosing a random number of structural panels.
        panelChoiceRange = np.arange(2*numPanels-1) # panel choice * 2 for each side of panel.
                                                    # -1 because no comps can go on the bottom side where it attaches to the lv
        sideOnPanelRange = np.arange(6) # choose which side goes on the panel. 6 for the 6 sides of a cuboid

        # put in the lv adapter
        # dimensions: 1 by 1 to fit a 937 mm fairing as described in the falcon payload users guide
        structSol.append(1)
        structSol.append(1)

        # Location: At the bottom of the usable cube
        structSol.append(0)
        structSol.append(0)
        structSol.append(-1)

        # Orientation: no rotation because a plate with its normal in the z is the base state
        structSol.append(0)
        structSol.append(0)
        structSol.append(0)


        for panelIdx in range(numPanels-1): # -1 because one panel is taken for the lv adapter
            # Size Choice between 0 and 2 meters
            structSol.append(np.random.uniform(0,2))
            structSol.append(np.random.uniform(0,2))

            # Location Choice
            structSol.append(np.random.uniform(-1,1))
            structSol.append(np.random.uniform(-1,1))
            structSol.append(np.random.uniform(-1,1))

            # Orientation Choice
            structSol.append(np.random.uniform(0,2*np.pi))
            structSol.append(np.random.uniform(0,2*np.pi))
            structSol.append(np.random.uniform(0,2*np.pi))

            # Used for testing
            # structSol.append(1)
            # structSol.append(1)
            # structSol.append(0)
            # structSol.append(0)
            # structSol.append(0)
            # structSol.append(np.pi/4)
            # structSol.append(0)
            # structSol.append(0)


        for compIdx in range(compLength):
            # Panel Choice
            compSol.append(np.random.choice(panelChoiceRange))

            # Location Choice
            compSol.append(np.random.uniform(-1,1))
            compSol.append(np.random.uniform(-1,1))

            # Orientation Choice
            compSol.append(np.random.choice(sideOnPanelRange))
            compSol.append(np.random.uniform(0,2*np.pi))

        solution = {"Structure":structSol, "Components":compSol}
        allSolutions.append(solution)

        costList, constraint = randSearchCostCalc(components,solution)
        allCosts.append(costList)
        allConstraints.append(constraint)

    tempCosts = []
    avgCosts = []
    allCosts = np.array(allCosts)
    maxCosts = (np.max(allCosts, axis=0)+1e-6)*1.1
    allCostsNorm = allCosts/maxCosts
    HVgrid = HypervolumeGrid([1,1,1,1,1])
    for ind,cost in enumerate(allCostsNorm):
        if not allConstraints[ind]:
            HVgrid.updateHV(cost,allSolutions[ind])
        allHV.append(HVgrid.getHV())
        tempCosts.append(cost)
        if (ind+1)%numBatches == 0:
            avgCosts.append(-np.mean(np.array(tempCosts),0))
            tempCosts = []
    
    return numRuns,allHV,HVgrid,avgCosts,maxCosts

### UNUSED CODE

# def optimization(components,structPanelList,maxCostList,date_str,optMethod,params):
#     # Minimize the cost of the configuration
#     global compList
#     global maxCosts
#     global structPanels
#     compList = components
#     maxCosts = maxCostList
#     structPanels = structPanelList

#     if optMethod == "GA":
#         num_generations,allHV,HVgrid,avgCosts = GAOptimization(components,structPanels,params)

#     # elif optMethod == "RL":
#     #     num_generations,allHV,HVgrid,avgCosts = RLOptTransformer.run(components,structPanels,maxCostList,date_str,params)

#     elif optMethod == "RS":
#         num_generations,allHV,HVgrid,avgCosts = randomSearch(components,structPanelList,maxCostList,params)

#     return num_generations,allHV,HVgrid,avgCosts

# def findLocalGradient(dimensions,locations,typeList,massList):
#     # Look at the cost change in each direction to find the local gradient
#     dirChange = [[0.01,0,0],[-0.01,0,0],[0,0.01,0],[0,-0.01,0],[0,0,0.01],[0,0,-0.01]]
#     elGrads = []
#     for i in range(len(dimensions)):
#         costList = []
#         elLoc = locations[i]
#         for dir in dirChange:
#             newLocations = list(locations)
#             newLoc = [loc+change for loc,change in zip(elLoc, dir)]
#             newLocations[i] = newLoc
#             newCost = sum(getCostComps(dimensions,newLocations,typeList,massList)) # Doesn't work but I forget why. If you ever want to use this, figure it out or uncomment the getCostParams function
#             costList.append(newCost)
#         xDeriv = (costList[0]-costList[1])/0.02
#         yDeriv = (costList[2]-costList[3])/0.02
#         zDeriv = (costList[4]-costList[5])/0.02
#         elGrads.append([xDeriv,yDeriv,zDeriv])
#     return elGrads

# def gradientOptimization(components):
    # elLocs = []
    # elDims = []
    # typeList = []
    # massList = []
    # for comp in components:
    #     elLocs.append(comp.location)
    #     elDims.append(comp.dimensions)
    #     typeList.append(comp.type)
    #     massList.append(comp.mass)
    # numComps = len(components)

    # allCost = []

    # # Gradient Based
    # costList = getCostComps(components)
    # cost = sum(costList) # Summing Cost to get single objective to optimize on
    # for costVal in costList:
    #     allCost.append([-costVal])
    # delCost = 1
    # threshold = 10**-2
    # print(cost)
    # allLocs = [elLocs]
    # numSteps = 0

    # numSteps = 0
    # while delCost > threshold:
    #     # Calculate the local gradient
    #     gradient = findLocalGradient(elDims,allLocs[-1],typeList,massList)
    #     newLocs = []

    #     # Get the required move
    #     for i in range(numComps):
    #         move = np.array(gradient[i])

    #         # Inverse Proportional to number of steps -- Really good -- Creates very compact designs
    #         if np.linalg.norm(move) == 0:
    #             newLocEl = allLocs[-1][i]
    #         else:
    #             moveNormalized = move/np.linalg.norm(move)
    #             oldLoc = np.array(allLocs[-1][i])
    #             newLocEl = np.ndarray.tolist(oldLoc + moveNormalized*-1/(numSteps+5))

    #         # Proportional to Gradient -- Jumpy and slow -- not good
    #         # oldLoc = np.array(allLocs[-1][i])
    #         # newLocEl = np.ndarray.tolist(oldLoc + move*-.0002)

    #         # Inverse Proportional to Gradient -- Always Blows up or doesn't move
    #         # oldLoc = np.array(allLocs[-1][i])
    #         # newLocEl = np.ndarray.tolist(oldLoc + 1/move*-.02)

    #         # Constant Step -- Works pretty well -- not so compact designs
    #         # if np.linalg.norm(move) == 0:
    #         #     newLocEl = allLocs[-1][i]
    #         # else:
    #         #     moveNormalized = move/np.linalg.norm(move)
    #         #     oldLoc = np.array(allLocs[-1][i])
    #         #     newLocEl = np.ndarray.tolist(oldLoc + moveNormalized*-.01)
    #         newLocs.append(newLocEl)

    #     # Add the newLoc into allLocs
    #     allLocs.append(newLocs)
    #     for i in range(len(components)):
    #         components[i].location = newLocs[i]
    #     newCostList = getCostComps(components)
    #     newCost = sum(newCostList) # Summing Cost to get single objective to optimize on
    #     for i in range(len(newCostList)):
    #         allCost[i].append(-newCostList[i])        
    #     delCost = abs(newCost-cost)
    #     cost = newCost
    #     print(newCost)
    #     numSteps+=1
    
    # for cost in allCost:
    #     plt.plot(cost)
    # plt.legend(["1000*overlapCostVal", "cmCostCalVal", "offAxisInertia", "onAxisInertia", "wireCostVal"])
    # plt.show()
    # print(numSteps)
    # return allLocs,numSteps

    # def GAFitnessFuncNR(GAInstance,solution,solutionIDX):
#     # Fitness function for pyGAD. Calls the cost function and inverts the output
#     comps = deepcopy(compList)
#     for i in range(len(comps)):
#         comps[i].location = [solution[3*i],solution[3*i+1],solution[3*i+2]]
#     # for i in range(len(comps)):
#     #     comps[i].location = [solution[4*i],solution[4*i+1],solution[4*i+2]]
#     #     dims = comps[i].dimensions
#     #     if solution[4*i+3] == 2:
#     #         dims = [dims[0],dims[2],dims[1]]
#     #     elif solution[4*i+3] == 3:
#     #         dims = [dims[1],dims[0],dims[2]]
#     #     elif solution[4*i+3] == 4:
#     #         dims = [dims[1],dims[2],dims[0]]
#     #     elif solution[4*i+3] == 5:
#     #         dims = [dims[2],dims[0],dims[1]]
#     #     elif solution[4*i+3] == 6:
#     #         dims = [dims[2],dims[1],dims[0]]
#     #     comps[i].dimensions = dims
            
#     costList = getCostComps(comps)
#     rewardList = []
#     for cost in costList:
#         rewardList.append(-cost)

#     global NFE
#     global allCosts
#     NFE+=1
#     allCosts.append(rewardList)

#     return rewardList

# def on_generationNR(ga_instance):
#     global last_fitness
#     print(f"Generation = {ga_instance.generations_completed}")
#     print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
#     print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
#     last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

# def GAOptimizationNR(components):
#     # uses a GA to find the optimal spacecraft configuration
#     # Parameters
#     num_generations = 10 # Number of generations.
#     num_parents_mating = 32 # Number of solutions to be selected as parents in the mating pool.

#     sol_per_pop = 128 # Number of solutions in the population.
#     num_genes = len(components)*3 # Without Rotations
#     # num_genes = len(components)*4 # With Rotations
#     save_best_solutions = True
#     init_range_low = -1
#     init_range_high = 1
#     # gene_space = [None] * num_genes
#     # for i in range(int(len(gene_space)/4)):
#     #     gene_space[(i+1)*4-1] = [1,2,3,4,5,6]
#     parent_selection_type = "nsga2"

#     global last_fitness
#     last_fitness = 0

#     global NFE
#     global allCosts
#     NFE = 0
#     allHV = []
#     allCosts = []
#     maxHV = 0

#     ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        save_best_solutions=save_best_solutions,
#                        init_range_low=init_range_low,
#                        init_range_high=init_range_high,
#                     #    gene_space=gene_space,
#                        parent_selection_type=parent_selection_type,
#                        fitness_func=GAFitnessFuncNR,
#                        on_generation=on_generationNR)
    
#     # Running the GA to optimize the parameters of the function.
#     ga_instance.run()

#     # ga_instance.plot_fitness(label=["100*overlapCostVal", "10*cmCostCalVal", "offAxisInertia", "0.1*onAxisInertia", "wireCostVal"], 
#     #                          title= "GA - Best Fitness for Each Generation")

#     allCostsnp = np.array(allCosts)
#     # maxCostList = getMaxCosts(components)

#     metric = Hypervolume(ref_point=np.array([100,1,1,1,1]))
#     hv = [metric.do(-point) for point in allCostsnp]
#     for h in hv:
#         if h > maxHV:
#             maxHV = h
#         allHV.append(maxHV)

#     # plt.plot(allHV)
#     # plt.show()
#     # negCost = np.array(negCost)
#     # for i in range(len(negCost[0])):
#     #     plt.plot(range(NFE),negCost[:,i])
#     # plt.legend(["1000*overlapCostVal", "cmCostCalVal", "offAxisInertia", "onAxisInertia", "wireCostVal"])
#     # plt.show()

#     # solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
#     allSolutions = ga_instance.best_solutions
#     allLocs = []
#     allDims = []
#     allRots = []
#     for sol in allSolutions:
#         compLocs = []
#         dims = []
#         rotList = []
#         for comp in components:
#             dims.append(comp.dimensions)
#         for i in range(int(len(sol)/3)):
#             compLocs.append([sol[3*i],sol[3*i+1],sol[3*i+2]])
#         # for i in range(int(len(sol)/4)):
#         #     compLocs.append([sol[4*i],sol[4*i+1],sol[4*i+2]])
#         #     if int(sol[4*i+3]) == 2:
#         #         dims[i] = [dims[i][0],dims[i][2],dims[i][1]]
#         #     elif int(sol[4*i+3]) == 3:
#         #         dims[i] = [dims[i][1],dims[i][0],dims[i][2]]
#         #     elif int(sol[4*i+3]) == 4:
#         #         dims[i] = [dims[i][1],dims[i][2],dims[i][0]]
#         #     elif int(sol[4*i+3]) == 5:
#         #         dims[i] = [dims[i][2],dims[i][0],dims[i][1]]
#         #     elif int(sol[4*i+3]) == 6:
#         #         dims[i] = [dims[i][2],dims[i][1],dims[i][0]]
#         #     rotList.append(int(sol[4*i+3]))
#         allLocs.append(compLocs)
#         allDims.append(dims)
#         # allRots.append(rotList)
        
#     return allLocs,allDims,num_generations,allHV

# def randomSearch(components, structPanels, maxCostList, params):
#     # Random Search
#     numRuns = params[0]*params[1] # batch size * epochs
#     numBatches = params[0]
#     desLength = len(components)
#     numPanels = len(structPanels)
#     HVgrid = HypervolumeGrid([1,1,1,1,1]) # Only 5 to eliminate constraint (overlap cost) from HV calculation

#     allHV = []
#     avgCosts = []
#     tempCosts = []

#     # variable ranges
#     panelChoiceRange = np.arange(2*numPanels) # panel choice * 2 for each side of panel
#     orientationRange = np.arange(24) # for orientation
#     locRange = np.linspace(-1, 1, 51)
#     for i in range(numRuns):
#         solution = []
#         for j in range(desLength):
#             solution.append(np.random.choice(panelChoiceRange))
#             solution.append(np.random.choice(locRange))
#             solution.append(np.random.choice(locRange))
#             solution.append(np.random.choice(orientationRange))

#         costList = randSearchCostCalc(components,structPanels,maxCostList,HVgrid,solution)
#         HVCosts = costList[1:]
#         if costList[0] < 0.01:
#             HVgrid.updateHV(HVCosts,solution)
#             # print("\n\n Valid Solution Found ",i,"\n\n")
#         allHV.append(HVgrid.getHV())
#         tempCosts.append(costList)
#         if (i+1)%numBatches == 0:
#             avgCosts.append(-np.mean(np.array(tempCosts),0))
#             tempCosts = []
        

#     return numRuns,allHV,HVgrid,avgCosts