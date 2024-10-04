import numpy as np
import time

class HypervolumeGrid:

    def __init__(self,refPoint):
        if isinstance(refPoint,list):
            refPoint = np.array(refPoint)
        self.refPoint = refPoint
        self.initializeGrid()

    def initializeGrid(self):
        """
        Function to initialize the hypervolume grid with the reference point
        """
        self.gridSize = 21 # maybe change this to have different resolution closer to the best point
        self.HVMax = np.prod(self.refPoint)

        exp = np.arange(len(self.refPoint)-1,-1,-1)
        self.mult = np.power(self.gridSize,exp)

        self.numPoints = (self.gridSize)**len(self.refPoint)
        self.dominated = np.zeros(int(self.numPoints), dtype=bool)

        self.paretoFrontPoint = np.array([])
        self.paretoFrontSolution = np.array([])

    def updateHV(self,point,solution):
        """
        Function to update the hypervolume grid and pareto front with a new point and index. 
        Index for each call should be unique and able to reference the solution.
        """
        if isinstance(point,list):
            point = np.array(point)
        if isinstance(solution,list):
            solution = np.array(solution)
        pointCalc = self.refPoint - point
        gridPoint = np.floor(pointCalc/self.refPoint * (self.gridSize-1))
        pointIdx = int(np.dot(gridPoint,self.mult))

        if not self.dominated[pointIdx]:
            dominatedInd = np.arange(self.numPoints,dtype=int)
            newDominated = np.ones(self.numPoints, dtype=bool)
            for dim in range(len(self.refPoint)):
                dimList = np.floor_divide(np.mod(dominatedInd,self.mult[dim]*self.gridSize,dtype=int),self.mult[dim],dtype=int)
                fitDim = dimList <= gridPoint[dim]
                newDominated = np.logical_and(newDominated,fitDim, dtype=bool)
            
            self.updateParetoFront(point,solution)

            self.dominated = np.logical_or(self.dominated, newDominated, dtype=bool)

    def updateParetoFront(self,newPoint,solution):
        """
        Add a new cost to the pareto front if it is not dominated
        """
        isDominated = False
        if self.paretoFrontPoint.size:
            isDominated = np.any(np.all(self.paretoFrontPoint < newPoint, axis = 1))
            newDominated = np.all(self.paretoFrontPoint > newPoint, axis = 1)

            self.paretoFrontPoint = self.paretoFrontPoint[~newDominated]
            self.paretoFrontSolution = self.paretoFrontSolution[~newDominated]
        
        if isDominated == False:
            self.paretoFrontPoint = np.vstack([self.paretoFrontPoint, newPoint]) if self.paretoFrontPoint.size else np.array([newPoint])
            self.paretoFrontSolution = np.vstack([self.paretoFrontSolution, solution]) if self.paretoFrontSolution.size else np.array([solution])

    def filterParetoFront(self, objInd, value):   
        """
        Filter the pareto front by the index of the objective and the desired value
        Ex: objInd = 1, value = 0.1 will return all points where the second objective is less than 0.1
        """
        if self.paretoFrontSolution.size:
            mask = self.paretoFrontPoint[:,objInd] < value
            self.paretoFrontPoint = self.paretoFrontPoint[mask]
            self.paretoFrontSolution = self.paretoFrontSolution[mask]

    def getHV(self):
        return np.sum(self.dominated)/self.numPoints * self.HVMax
    

class ParetoFront:
    def __init__(self):
        self.paretoFront = np.array([])

    def updatePF(self, newCosts, return_mask = True):
        """
        Add a new cost to the pareto front if it is not dominated
        """
        isDominated = False
        if self.paretoFront.size:
            isDominated = np.any(np.all(self.paretoFront < newCosts, axis = 1))
            newDominated = np.all(self.paretoFront > newCosts, axis = 1)

            self.paretoFront = self.paretoFront[~newDominated]
        
        if isDominated == False:
            self.paretoFront = np.vstack([self.paretoFront, newCosts]) if self.paretoFront.size else np.array([newCosts])

        return self.paretoFront


def getParetoFront(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    costs = np.array(costs)
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

# class HypervolumeGrid:

#     def __init__(self,refPoint):
#         if isinstance(refPoint,list):
#             refPoint = np.array(refPoint)
#         self.refPoint = refPoint
#         self.initializeGrid()

#     def initializeGrid(self):
#         # Function to initialize the grid with the reference point
#         self.gridSize = 11
#         dims = np.zeros([len(self.refPoint),self.gridSize])
#         for i,a in enumerate(self.refPoint):
#             dims[i] = (np.linspace(0,a,self.gridSize))
        
#         mesh = np.meshgrid(*dims)
#         self.grid = np.vstack(list(map(np.ravel,mesh))).T
#         self.dominated = np.zeros(len(self.grid))

#     def updateGrid(self,point):
#         if isinstance(point,list):
#             point = np.array(point)
#         pointCalc = self.refPoint - point
#         if not self.isDominated(pointCalc):
#             newDominated = np.prod((self.grid <= pointCalc), axis=1)
#             self.dominated = np.logical_or(self.dominated, newDominated)
#         else:
#             print("Point is dominated")

#     def isDominated(self,pointCalc):
#         # Function to check if a point is dominated by the grid
#         gridPoint = np.floor(pointCalc/self.refPoint * (self.gridSize-1))
#         exp = np.arange(len(pointCalc)-1,-1,-1)
#         mult = np.power(self.gridSize,exp)
#         idx = np.dot(gridPoint,mult)
#         dom = self.dominated[int(idx)]
#         return dom

#     def getHV(self):
#         return np.sum(self.dominated)
