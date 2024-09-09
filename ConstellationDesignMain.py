from SpacecraftDesignSelectionConstellation import *
from CostEstimationJSON import loadJSONCostEstimation
from CallCoverageAnalysis import tatcCovReqTransformer

archPath = 'JsonFiles/arch.json'

scMasses, subsMasses, constComponents, costEstimationJSONFile = loadJSONConst(archPath)

totalMissionCosts = loadJSONCostEstimation(costEstimationJSONFile)
print("Total Mission Costs: ", totalMissionCosts)

# harmonicMeanRevisit = tatcCovReqTransformer(coverageRequestJSONFile)