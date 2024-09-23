from HypervolumeUtils import HypervolumeGrid, ParetoFront, getParetoFront
from pymoo.indicators.hv import HV
import numpy as np
import matplotlib.pyplot as plt
import time

costs = np.random.rand(200, 6)

t0 = time.time()
print("PyMoo With Pareto")
hvPymoo = HV(ref_point=[1, 1, 1, 1, 1, 1])

hvListPymoo = []

pf = ParetoFront()
for cost in costs:
    pf.updatePF(cost)

    hvListPymoo.append(hvPymoo(pf.paretoFront))

t1 = time.time()
print("Alex Without Pareto")
hvAlex = HypervolumeGrid([1, 1, 1, 1, 1, 1])

hvListAlex = []

for cost in costs:
    hvAlex.updateHV(cost)
    hvListAlex.append(hvAlex.getHV())

t2 = time.time()
print("PyMoo Without Pareto")
hvPymoo2 = HV(ref_point=[1, 1, 1, 1, 1, 1])

hvListPymoo2 = []

for i in range(len(costs)):

    hvListPymoo2.append(hvPymoo2(costs[:i+1]))

t3 = time.time()
print("Alex With Pareto")
hvAlex2 = HypervolumeGrid([1, 1, 1, 1, 1, 1])

hvListAlex2 = []

pf2 = ParetoFront()
for cost in costs:
    pf2.updatePF(cost)
    if (pf2.paretoFront[-1] == cost).all():
        hvAlex2.updateHV(cost)
    hvListAlex2.append(hvAlex2.getHV())

t4 = time.time()

print('PyMoo time Pareto: ', t1-t0)
print('Alex time No Pareto: ', t2-t1)
print('PyMoo time No Pareto: ', t3-t2)
print('Alex time Pareto: ', t4-t3)

# Plot the hypervolumes for all functions
plt.figure(figsize=(12, 12))

# Plot the hypervolumes for the functions with Pareto
plt.subplot(2, 1, 1)
plt.plot(hvListPymoo, color='blue', label='Hypervolume - PyMoo with Pareto')
plt.plot(hvListAlex2, color='orange', label='Hypervolume - Alex with Pareto')
plt.xlabel('Iteration')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Convergence with Pareto')
plt.legend()

# Plot the hypervolumes for the functions without Pareto
plt.subplot(2, 1, 2)
plt.plot(hvListAlex, color='red', label='Hypervolume - Alex without Pareto')
plt.plot(hvListPymoo2, color='green', label='Hypervolume - PyMoo without Pareto')
plt.xlabel('Iteration')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Convergence without Pareto')
plt.legend()

plt.tight_layout()
plt.show()

