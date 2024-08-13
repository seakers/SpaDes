import numpy as np
import matplotlib.pyplot as plt
import pygad
from ConfigurationCost import thermalCost
from SCDesignClasses import Component
from copy import deepcopy

def getCube(dimensions,location):
    # Function to transform cube dimensions and location into a form that plot_surface can plot
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta) * dimensions[0] + location[0]
    y = np.sin(Phi)*np.sin(Theta) * dimensions[1] + location[1]
    z = np.cos(Theta)/np.sqrt(2) * dimensions[2] + location[2]
    return x,y,z

def thermalCostTest(ga_instance, solution, solutionIDX):
    locations = []
    for i, comp in enumerate(componentList):
        locations.append(solution[i*3:i*3+3])

    tCost = thermalCost(dimensions,locations,heatDisps)
    return tCost

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

dimensions = []
heatDisps = []
for comp in componentList:
    dimensions.append(comp.dimensions)
    heatDisps.append(comp.heatDisp)

# def on_generation(ga_instance):
#     global last_fitness
#     print(f"Generation = {ga_instance.generations_completed}")
#     print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
#     print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
#     last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

# last_fitness = 0

# ga_instance = pygad.GA(num_generations=500,
#                     num_parents_mating=64,
#                     sol_per_pop=256,
#                     num_genes=3*len(componentList),
#                     init_range_low=-1,
#                     init_range_high=1,
#                     gene_space=[np.linspace(1,-1,51)]*3*len(componentList),
#                     save_best_solutions=True,
#                     fitness_func=thermalCostTest,
#                     on_generation=on_generation)

# # Running the GA to optimize the parameters of the function.
# ga_instance.run()

# ga_instance.plot_fitness()

# bestSols = ga_instance.best_solutions[-1]

# locations = []
# for i in range(len(componentList)):
#     locations.append([bestSols[i*3],bestSols[i*3+1],bestSols[i*3+2]])

locations = [[1,1,1]]*len(componentList)
altCorners = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]]
sortHeats = sorted(heatDisps)
heatCutoff = sortHeats[6]
for i,heat in enumerate(heatDisps):
    if heat <= heatCutoff:
        locations[i] = altCorners.pop()

tCost = thermalCost(dimensions,locations,heatDisps)

print("Locations: ",locations)
print("Heat Disps: ",heatDisps)
print("Thermal Cost: ",tCost)

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
    xGA,yGA,zGA = getCube(dimensions[i],locations[i])
    plot1 = ax1.plot_surface(xGA,yGA,zGA)
plt.title("Visualization of Configuration GA")

plt.show()