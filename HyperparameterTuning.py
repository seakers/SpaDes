import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ConfigurationOptimization import *
from SCDesignClasses import Component
from ConfigurationCost import maxCostComps
import datetime
import os
import itertools

def main():
    def getCube(dimensions,location):
        # Function to transform cube dimensions and location into a form that plot_surface can plot
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi)

        x = np.cos(Phi)*np.sin(Theta) * dimensions[0] + location[0]
        y = np.sin(Phi)*np.sin(Theta) * dimensions[1] + location[1]
        z = np.cos(Theta)/np.sqrt(2) * dimensions[2] + location[2]
        return x,y,z

    componentList = [
        Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=1.5),
        Component(type="solar panel", mass=1.6, dimensions=[.21,.52,.01], heatDisp=1.6),
        Component(type="solar panel", mass=1.4, dimensions=[.19,.49,.01], heatDisp=1.4),

        Component(type="payload", mass=6.5, dimensions=[.3,.24,.22], heatDisp=3.2),
        Component(type="payload", mass=5.5, dimensions=[.28,.22,.2], heatDisp=3),

        Component(type="transmitter", mass=3.8, dimensions=[.25,.1,.08], heatDisp=12),
        Component(type="transmitter", mass=4.0, dimensions=[.23,.12,.09], heatDisp=11),

        Component(type="receiver", mass=3.3, dimensions=[.21,.12,.05], heatDisp=9),
        Component(type="receiver", mass=3.5, dimensions=[.22,.13,.06], heatDisp=9.5),

        Component(type="antenna", mass=4.5, dimensions=[.35,.14,.12], heatDisp=9.7),
        Component(type="antenna", mass=3.2, dimensions=[.24,.1,.08], heatDisp=8),
        Component(type="antenna", mass=4.0, dimensions=[.34,.15,.1], heatDisp=9),

        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.3),
        Component(type="star tracker", mass=1.8, dimensions=[.12,.14,.11], heatDisp=1.4),
        Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.1], heatDisp=1.2),

        Component(type="sun sensor", mass=1.2, dimensions=[.1,.09,.08], heatDisp=0.9),
        Component(type="sun sensor", mass=1.1, dimensions=[.1,.08,.07], heatDisp=1),
        Component(type="sun sensor", mass=1.3, dimensions=[.12,.1,.09], heatDisp=1.1),

        Component(type="battery", mass=5.8, dimensions=[.23,.21,.13], heatDisp=2.2),
        Component(type="battery", mass=6.0, dimensions=[.24,.22,.14], heatDisp=2.4),

        Component(type="PCU", mass=7, dimensions=[.26,.2,.14], heatDisp=6.5),
        Component(type="PCU", mass=6.5, dimensions=[.25,.19,.13], heatDisp=6.3),

        Component(type="OBDH", mass=9, dimensions=[.24,.19,.16], heatDisp=5.8),
        Component(type="OBDH", mass=8.8, dimensions=[.23,.18,.15], heatDisp=5.7),

        Component(type="reaction wheel", mass=3, dimensions=[.14,.12,.1], heatDisp=3.2),
        Component(type="reaction wheel", mass=3.2, dimensions=[.15,.13,.11], heatDisp=3.3),

        Component(type="propellant tank", mass=13, dimensions=[.3,.25,.2], heatDisp=4.2),
        Component(type="propellant tank", mass=12.5, dimensions=[.29,.24,.19], heatDisp=4.1),

        Component(type="attitude thruster", mass=2.5, dimensions=[.15,.14,.12], heatDisp=4.8),
        Component(type="attitude thruster", mass=2.7, dimensions=[.16,.15,.13], heatDisp=5),

        Component(type="IMU", mass=2.5, dimensions=[.14,.12,.09], heatDisp=2.5),
        Component(type="IMU", mass=2.4, dimensions=[.13,.11,.08], heatDisp=2.4),

        Component(type="atomic clock", mass=1.8, dimensions=[.12,.11,.07], heatDisp=1.6),

        Component(type="heater", mass=1.2, dimensions=[.09,.07,.05], heatDisp=1.9),
        Component(type="heater", mass=1.3, dimensions=[.1,.08,.06], heatDisp=2),

        Component(type="gyro", mass=3.1, dimensions=[.19,.13,.02], heatDisp=2.8),
        Component(type="gyro", mass=3.0, dimensions=[.18,.12,.02], heatDisp=2.7),

        Component(type="magnetometer", mass=1.4, dimensions=[.1,.12,.1], heatDisp=1.1),
        Component(type="magnetometer", mass=1.5, dimensions=[.11,.13,.09], heatDisp=1.2),

        Component(type="accelerometer", mass=1.8, dimensions=[.13,.11,.09], heatDisp=2.2),
        Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.1)
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

    # Define hyperparameters and ranges. Just three for each that represent the standard range for now
    # Not worrying about making it iso-nfe
    # inspiration for hyperparameters and values is from https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    # minibatch = [16]
    # epochs = [50]
    # # clipping = [0.05,0.1,0.2]
    # clipping = [0.2, 0.3]
    # KL = [0.003,0.01,0.03]
    # # KL = [0.003,0.]
    # # gamma = [0.95,0.99,0.999]
    # gamma = [0.999]
    # # lam = [0.9,0.95,0.99]
    # lam = [0.95, 0.99]
    # # lr = [0.0003,0.001,0.003]
    # # lr = [0.001, 0.003, 0.01]
    # lr = [0.003, 0.01]
    # iterations = [5, 10]

    # minibatch = [8]
    # epochs = [10]
    # clipping = [0.1]
    # KL = [0.01]
    # gamma = [0.99]
    # lam = [0.95]
    # lr = [0.001]
    # iterations = [5]

    minibatch = 16 # batch size/4 (b/c 4 threads)
    epochs = 50
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

    # calculate here so it is only done once (for speed)
    maxCostList = maxCostComps(componentList,structPanelList)

    # Optimize
    numRuns = 1

    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'ResultGraphs/{date_str}', exist_ok=True)

    # Genetic Algorithm
    allHVGA = []
    allAvgCostsGA = []
    allHVRand = []

    print("\n\n\nRUN: GA\n\n")
    numStepsGA, allHVGARun, HVgridGA, avgCostsGA = optimization(componentList,structPanelList,maxCostList,date_str,"GA",params)
    allHVGA.append(allHVGARun)

    print("\n\n\nRUN: Random Search\n\n")
    numStepsRand, allHVRandRun, HVgridRand, avgCostsRand = optimization(componentList,structPanelList,maxCostList,date_str,"rand",params)
    allHVRand.append(allHVRandRun)

    allHVGA = np.array(allHVGA)
    allHVRand = np.array(allHVRand)

    medianHVGA = np.median(allHVGA,0)
    medianHVRand = np.median(allHVRand,0)

    plt.figure()
    plt.plot(medianHVGA, label="GA Hypervolume")
    plt.plot(medianHVRand, label="Random Search Hypervolume")
    plt.legend(loc="lower right")
    plt.ylim(.5, .7)
    plt.yticks(np.arange(.5, .71, 0.01))
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.title(f"GA vs Random Search Hypervolume")
    plt.savefig(f"ResultGraphs/{date_str}/GA_RandomSearch_HypervolumeComparisonTransformer.png")

    # # Reset locations and dimensions for Reinforcement Learning
    # i = 0
    # for comp in componentList:
    #     comp.location = compLocs[i]
    #     comp.dimensions = compDims[i]
    #     i+=1

    # allHVRL = []
    # allAvgCostsRL = []
    # fullFactEnum = itertools.product(minibatch,epochs,clipping,KL,gamma,lam,lr,iterations)
    # for params in fullFactEnum:
    #     t0 = time.time()
    #     paramString = f"{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{params[6]}_{params[7]}"
    #     print("\n\n\nPARAMS\n")
    #     print("Minibatch: ",params[0],"\nEpochs: ",params[1],"\nClipping: ",params[2],"\nKL: ",params[3],"\nGamma: ",params[4],"\nLambda: ",params[5],"\nLearning Rate: ",params[6],"\nIterations: ",params[7],"\n\n")

    #     numStepsRL, allHVRL, HVgridRL, avgCostsRL = optimization(componentList,structPanelList,maxCostList,date_str,"RL",params)
    #     # allHVRL.append(allHVRLRun)
    #     finalHVRL = HVgridRL.getHV()
    #     allHVRL.append(finalHVRL)

    #     allHVRL = np.array(allHVRL)

    #     # medianHVRL = np.median(allHVRL,0)
    #     # q1HVRL = np.quantile(allHVRL,.25,axis=0)
    #     # q3HVRL = np.quantile(allHVRL,.75,axis=0)
    #     # maxHVRL = np.max(allHVRL,0)
    #     # minHVRL = np.min(allHVRL,0)

    #     plt.figure()
    #     plt.plot(allHVRL)
    #     # plt.plot(maxHVRL,linestyle='dashed')
    #     # plt.plot(minHVRL,linestyle='dotted')
    #     # plt.fill_between(range(len(medianHVRL)), q1HVRL, q3HVRL, alpha=.5, linewidth=0)
    #     plt.legend(["Hypervolume"],loc="lower right")
    #     plt.ylim(.5, .7)
    #     plt.yticks(np.arange(.5, .71, 0.01))
    #     plt.xlabel("Number of Function Evaluations")
    #     plt.ylabel("Hypervolume")
    #     plt.title(f"Deep RL {paramString} Hypervolume {finalHVRL}")
    #     plt.savefig(f"ResultGraphs/{date_str}/{paramString}_HypervolumeComparisonTransformer{finalHVRL}.png")
    #     print("Time taken: ",time.time()-t0)


if __name__ == "__main__":
    main()