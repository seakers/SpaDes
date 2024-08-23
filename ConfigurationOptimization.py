import numpy as np
import pygad
from copy import deepcopy
from ConfigurationCost import *
from RLOpt import RLWrapper
from ConfigUtils import getOrientation
from HypervolumeUtils import HypervolumeGrid

def GAFitnessFunc(GAInstance,solution,solutionIDX):
    # Fitness function for pyGAD. Calls the cost function and inverts the output
    global structPanels
    comps = deepcopy(compList)
    surfNormal = np.array([0,0,1])

    for i in range(len(comps)):
        transMat = getOrientation(int(solution[4*i+3]))
        comps[i].orientation = transMat
    
        panelChoice = structPanels[int(solution[4*i]%len(structPanels))]
        if solution[4*i] >= len(structPanels):
            surfNormal = surfNormal * -1
        
        surfLoc = np.matmul(panelChoice.orientation,np.multiply([solution[4*i+1],solution[4*i+2],surfNormal[2]],np.array(panelChoice.dimensions)/2))
        comps[i].location = surfLoc + np.multiply(np.abs(np.matmul(transMat,np.array(comps[i].dimensions)/2)),np.matmul(panelChoice.orientation,surfNormal)) + panelChoice.location
            
    costList = getCostComps(comps,structPanels,maxCosts)
    rewardList = -np.array(costList)

    global NFE
    global allCosts
    global HVgrid
    global allHV
    NFE+=1
    allCosts.append(rewardList)
    HVgrid.updateHV(costList,solution)
    allHV.append(HVgrid.getHV())

    return rewardList

def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

def GAOptimization(components,structPanels):
    # uses a GA to find the optimal spacecraft configuration
    # Parameters
    num_generations = 750 # Number of generations.
    num_parents_mating = 32 # Number of solutions to be selected as parents in the mating pool.

    sol_per_pop = 128 # Number of solutions in the population.
    num_genes = len(components)*4 # With Rotations
    save_best_solutions = True
    gene_space = [None] * num_genes

    numPanels = len(structPanels)
    for i in range(num_genes):
        if i%4 == 0:
            gene_space[i] = np.arange(2*numPanels) # panel choice * 2 for each side of panel
        elif i%4 == 3:
            gene_space[i] = np.arange(24) # for orientation
        else:
            gene_space[i] = np.linspace(-1, 1, 51)

    parent_selection_type = "nsga2"

    global last_fitness
    last_fitness = 0

    global NFE
    global allCosts
    global HVgrid
    global allHV

    NFE = 0
    allHV = []
    allCosts = []
    HVgrid = HypervolumeGrid([1,1,1,1,1,1])
    allHV = []

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       save_best_solutions=save_best_solutions,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       fitness_func=GAFitnessFunc,
                       on_generation=on_generation)
    
    # Running the GA to optimize the parameters of the function.
    ga_instance.run()


    pfSolutions = HVgrid.paretoFrontSolution
    pfCosts = HVgrid.paretoFrontPoint
        
    return num_generations,allHV,pfSolutions,pfCosts

def optimization(components,structPanelList,maxCostList,optMethod):
    # Minimize the cost of the configuration
    global compList
    global maxCosts
    global structPanels
    compList = components
    maxCosts = maxCostList
    structPanels = structPanelList

    if optMethod == "GA":
        num_generations,allHV,pfSolutions,pfCosts = GAOptimization(components,structPanels)

    elif optMethod == "RL":
        num_generations,allHV,pfSolutions,pfCosts = RLWrapper.run(components,structPanels,maxCostList)

    return num_generations,allHV,pfSolutions,pfCosts

### UNUSED CODE

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