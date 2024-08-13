import numpy as np
import time

class HypervolumeGrid:

    def __init__(self,refPoint):
        if isinstance(refPoint,list):
            refPoint = np.array(refPoint)
        self.refPoint = refPoint
        self.initializeGrid()

    def initializeGrid(self):
        # Function to initialize the grid with the reference point
        gridSize = 21
        dims = np.zeros([len(self.refPoint),gridSize])
        for i,a in enumerate(self.refPoint):
            dims[i] = (np.linspace(0,a,gridSize))
        
        mesh = np.meshgrid(*dims)
        self.grid = np.vstack(list(map(np.ravel,mesh))).T
        self.dominated = np.zeros(len(self.grid))

    def updateGrid(self,point):
        if isinstance(point,list):
            point = np.array(point)
        pointCalc = self.refPoint - point
        newDominated = np.prod((self.grid <= pointCalc), axis=1)
        self.dominated = np.logical_or(self.dominated, newDominated)

    def getHV(self):
        return np.sum(self.dominated)

hvg = HypervolumeGrid([100,1,1,1,1,1])
points = np.random.rand(5,6)
points = np.multiply(points,np.array([100,1,1,1,1,1]))
# points = [[0,0,0,0,0,0]]
for point in points:
    print(point)
    hvg.updateGrid(point)
    HV = hvg.getHV()
    print(HV)