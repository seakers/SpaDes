import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ConfigurationCostStruct import *
import scipy.signal
from ConfigUtils import *
from HypervolumeUtils import HypervolumeGrid, mergeHV
import time
from TransformerArchitectureStruct import *
from SCDesignClasses import *
from DDPsetup import setup, cleanup
import os
import datetime
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle

def trainsformerRLTraining(components,maxCosts,date_str,params):
    world_size = torch.cuda.device_count()
    print("Available GPUs:", torch.cuda.device_count())

    mp.spawn(run_ddp, args=(world_size, components, maxCosts, date_str, params), nprocs=world_size, join=True)
    
    # Load saved data
    NFE, allDes, allCosts, avgCosts = load_results(world_size)
    HVgrid = HypervolumeGrid([1,1,1,1,1])
    allHV = []
    for i in range(len(allDes)):
        if isinstance(allDes[i],dict):
            HVgrid.updateHV(allCosts[i], allDes[i])
        allHV.append(HVgrid.getHV())

    return NFE, allHV, HVgrid, avgCosts
    
def load_results(world_size):
    NFE = world_size * np.load(f'NFE0.npy')
    avgCosts = []
    allDes = []
    allCosts = []
    for rank in range(world_size):
        avgCosts.append(np.load(f'avgCosts{rank}.npy', allow_pickle=True))
        allDes.append(np.load(f'allDes{rank}.npy', allow_pickle=True))
        allCosts.append(np.load(f'allCosts{rank}.npy', allow_pickle=True))

    avgCostsflat = []
    for i in range(len(avgCosts[0])):
        tempAvgCosts = []
        for sublist in avgCosts:
            tempAvgCosts.append(sublist[i])
        avgCostsflat.append(np.mean(tempAvgCosts,0))

    allDesflat = []
    for i in range(len(allDes[0])):
        for sublist in allDes:
            allDesflat.append(sublist[i])

    allCostsflat = []
    for i in range(len(allCosts[0])):
        for sublist in allCosts:
            allCostsflat.append(sublist[i])

    return NFE, allDesflat, allCostsflat, avgCostsflat

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def run_ddp(rank, world_size, components, maxCosts, date_str, params):

    setup(rank, world_size)
    params[0] = int(params[0]/world_size)

    # Training code here
    run_training(rank, world_size, components, maxCosts, date_str, params)

    # Always cleanup even if an error occurs
    cleanup()


def run_training(rank, world_size, components, maxCosts, date_str, params):
    # Environment setup
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['NCCL_DEBUG'] = 'INFO'

    device = torch.device(f'cuda:{rank}')
    # print("Device is: ", device)
    # print("Rank is: ", rank)
    torch.cuda.set_device(device)
    epochs = params[1]
    num_components = len(components)

    # Initialize DDP models
    actor, critic = get_models(num_components, device, params, date_str)

    actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[rank], output_device=rank)
    critic = torch.nn.parallel.DistributedDataParallel(critic, device_ids=[rank], output_device=rank)

    NFE = 0
    allCLoss = []
    allLoss = []
    allKL = []
    avgCosts = []
    allDes = []
    allCosts = []

    for x in range(epochs):
        print("Epoch: ", x, "Rank: ", rank)
        
        actor, critic, NFE, allDes, allCosts, avgCost, c_loss, loss, kl = run_epoch(
            actor, critic, components, NFE, maxCosts, allDes, allCosts, device, rank, params
        )

        allCLoss.append(c_loss)
        allLoss.append(loss)
        allKL.append(kl)
        avgCosts.append(avgCost)


    torch.save(actor.module.state_dict(), f"ResultGraphs/{date_str}/actorTransformer.pth")
    torch.save(critic.module.state_dict(), f"ResultGraphs/{date_str}/criticTransformer.pth")

    # Plotting and saving results (only rank 0 to avoid duplication)
    if rank == 0:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(range(epochs), allCLoss, label='C Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(range(epochs), allLoss, label='Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Actor Loss')
        plt.title('Actor Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(range(epochs), allKL, label='KL Divergence', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence vs Epochs')
        plt.grid()
        plt.legend()

        plt.tight_layout()

        os.makedirs(f'ResultGraphs/{date_str}', exist_ok=True)
        plt.savefig(f'ResultGraphs/{date_str}/RLTraining.png')

    # Save Results in files

    print('Finished')
    # Save results to files for later access
    np.save(f'NFE{rank}.npy', NFE)
    np.save(f'allCosts{rank}.npy', allCosts)
    np.save(f'allDes{rank}.npy', allDes)
    np.save(f'avgCosts{rank}.npy', avgCosts)

def get_models(num_components, device, params, date_str):
    actor = Actor(num_components=num_components, device=device, params=params)
    critic = Critic(num_components=num_components, device=device, params=params)

    actor.to(device)
    critic.to(device)

    # if os.path.exists(f"ResultGraphs/{date_str}/actorTransformer.pth") and os.path.exists(f"ResultGraphs/{date_str}/criticTransformer.pth"):
    #     print("\n\n Transfer Learning from Existing Models:\n\n")
    #     actor.load_state_dict(torch.load(f"ResultGraphs/{date_str}/actorTransformer.pth"))
    #     critic.load_state_dict(torch.load(f"ResultGraphs/{date_str}/criticTransformer.pth"))

    inputs = torch.zeros(size=(1,num_components*4)).to(device)

    actor(inputs)
    critic(inputs)

    return actor, critic

# def get_existing_models(num_components, num_panels, device):
#     actor = Actor(num_components=num_components, num_panels=num_panels, device=device)
#     critic = Critic(num_components=num_components, device=device)

#     actor.to(device)
#     critic.to(device)

#     actor.load_state_dict(torch.load('actor.pth'))
#     critic.load_state_dict(torch.load('critic.pth'))

#     inputs = torch.zeros(size=(1,num_components*4)).to(device)

#     actor(inputs)
#     critic(inputs)

#     return actor, critic

def run_epoch(actor, critic, components, NFE, maxCosts, allDes, allCosts, device, rank, params):

    mini_batch_size = params[0]
    numComponents = len(components)
    num_actions = 9*9 + 5*numComponents 
    
    rewards = [[] for x in range(mini_batch_size)]
    actions = [[] for x in range(mini_batch_size)]
    logprobs = [[] for x in range(mini_batch_size)]
    designs = [[] for x in range(mini_batch_size)]

    observation = [[] for x in range(mini_batch_size)]
    critic_observations = [[] for x in range(mini_batch_size)]

    num_panels = np.zeros(mini_batch_size)-1

    face_choice_norm = np.linspace(0,1,6) # only discrete variable which needs transformation.

    # Create a set of six random weights
    weightsNonNorm = np.random.rand(mini_batch_size,5)
    weights = weightsNonNorm / weightsNonNorm.sum(axis=1, keepdims=True)

    panelActs = np.tile(np.arange(9),9) # 8 normal actions plus choice to have a new panel. 9 times for 9 panels
                                        # really 10 minus one because one is set to be the lv adapter
    compActs = np.tile(np.arange(5),numComponents) # five actions per component


    # 1. Sample actor
    # for the panels I wanted to only have it generate as long as it wants another panel, but to take advantage of 
    # parallelization I decided to do it this way. Anything generated for panels after it generates a 0 for the choice
    # to make another panel will be ignored in the design objective calculation.
    panelCounter = np.zeros(mini_batch_size)
    for act in panelActs:
        log_probs, sel_actions = actor.module.sample_configuration(observation, 'panel', act)
        log_probs = log_probs.tolist()
        sel_actions = sel_actions.tolist()

        for idx, action in enumerate(sel_actions):
            if act == 0: # choice to create another panel
                panelCounter[idx] = panelCounter[idx] + 1 # minimum is one because of the lv adapter
                designs[idx].append(action)
                observation[idx].append(action) # should be 0 or 1
                if num_panels[idx] == -1 and action == 0: # action=0 means no more panels
                    num_panels[idx] = panelCounter[idx]
            elif act == 1: # xdim
                dim = action*2 # to go from 0 to 2
                designs[idx].append(dim)
                observation[idx].append(action)
            elif act == 2: # ydim
                dim = action*2 # to go from 0 to 2
                designs[idx].append(dim)
                observation[idx].append(action)
            elif act == 3: # xloc
                loc = action*2-1 # to go from -1 to 1
                designs[idx].append(loc)
                observation[idx].append(action)
            elif act == 4: # yloc
                loc = action*2-1 # to go from -1 to 1
                designs[idx].append(loc)
                observation[idx].append(action)
            elif act == 5: # zloc
                loc = action*2-1 # to go from -1 to 1
                designs[idx].append(loc)
                observation[idx].append(action)
            elif act == 6: # phi
                rot = action*2*np.pi # to go from 0 to 2pi
                designs[idx].append(rot)
                observation[idx].append(action)
            elif act == 7: # theta
                rot = action*2*np.pi # to go from 0 to 2pi
                designs[idx].append(rot)
                observation[idx].append(action)
            elif act == 8: # psi
                rot = action*2*np.pi # to go from 0 to 2pi
                designs[idx].append(rot)
                observation[idx].append(action)

            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            
            rewards[idx].append(0)
            
    for i in range(len(num_panels)):
        if num_panels[i] == -1:
            num_panels[i] = 10

    for act in compActs:
        log_probs, sel_actions = actor.module.sample_configuration(observation, 'comp', act)

        log_probs = log_probs.tolist()
        sel_actions = sel_actions.tolist()
        for idx, action in enumerate(sel_actions):
            if act == 0: # panel choice
                panelChoices = 2*num_panels[idx]-1 # bin the continuous variable to make it discrete
                bins = (np.arange(panelChoices) + 1)/panelChoices
                panelChoice = np.digitize(action,bins,right=True)
                designs[idx].append(panelChoice)
                observation[idx].append(action)
            elif act == 1: # xloc
                loc = action*2-1 # to go from -1 to 1
                designs[idx].append(loc)
                observation[idx].append(action)
            elif act == 2: # yloc
                loc = action*2-1 # to go from -1 to 1
                designs[idx].append(loc)
                observation[idx].append(action)
            elif act == 3: # face Choice
                designs[idx].append(action)
                observation[idx].append(face_choice_norm[action])
            elif act == 4: # rotation choice
                rot = action*2*np.pi # to go from 0 to 2pi
                designs[idx].append(rot)
                observation[idx].append(action)

            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            
            rewards[idx].append(0) # just zero here, we will put in intermediate rewards in the next section

    # Post processing
    # - transform flattened design to configuration
    # - evaluate configuration
    # - record reward    
    cost = []
    firstPanel = [1,1,0,0,-1,0,0,0] # lv adapter panel
    for idx, des in enumerate(designs):
        structSol = np.zeros(int(num_panels[idx]*8))
        structSol[:8] = firstPanel

        if num_panels[idx] > 1:
            structDes = np.array(des[:int((num_panels[idx]-1)*9)])
            structDesMask = np.ones(len(structDes), dtype=bool)
            structDesMask[0::9] = False
            structDes = structDes[structDesMask] # get rid of all the stopping criteria
            try:
                structSol[8:] = structDes
            except:
                print("ERROR")
                print(num_panels[idx])
                print(len(structDes))

        compSol = np.array(des[-5*numComponents:])

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

            # put in intemediate constraint rewards
            intermediateConstraint = overlapCostSingleSAT(structPanels, components[:i], components[i])
            if intermediateConstraint: # if it is violated True = violated
                rewards[idx][-5*(numComponents-i) + 5] = -100

        costVals, constraint = getCostComps(components,structPanels,maxCosts)
        NFE += 1
        adjCostVals = -np.array(costVals)

        if not constraint:
            allDes.append(solutionDict)
            allCosts.append(-adjCostVals)
        else:
            allDes.append(-np.ones(len(solutionDict)))
            allCosts.append(-np.ones(len(adjCostVals)))
        #     HVgrid.updateHV(HVCosts, des)
        # allHV.append(HVgrid.getHV())

        rewards[idx][-1] = rewards[idx][-1] + np.dot(weights[idx],adjCostVals)
        cost.append(adjCostVals)

    # t3 = time.time()
    # print("Time to evaluate: ", t3-t2)

    # Sample Critic
    critic_values = []
    for action_idx in range(num_actions):
        critic_observations = []
        for idx in range(mini_batch_size):
            obs = observation[idx]
            critic_obs = []
            critic_obs.extend(obs[:action_idx + 1])
            critic_observations.append(critic_obs)
        crit_vals = critic.module.sample_critic(critic_observations)
        crit_vals = crit_vals.detach().cpu().numpy()
        critic_values.append(np.sum(np.multiply(-crit_vals,weights),1))

    # t4 = time.time()
    # print("Time to sample critic: ", t4-t3)
    
    values = [[] for x in range(mini_batch_size)]
    for act_idx, act_vals in enumerate(critic_values):
        for batch_idx, val in enumerate(act_vals):
            values[batch_idx].append(val)

    for idx in range(mini_batch_size):
        values[idx].append(values[idx][-1])

    # t5 = time.time()
    # print("Time to process values: ", t5-t4)
    

    gamma = params[4]
    lam = params[5]
    all_advantages = [[] for x in range(mini_batch_size)]
    all_returns = [[] for x in range(mini_batch_size)]
    for idx in range(mini_batch_size):
        d_reward = np.array(rewards[idx])
        d_value = np.array(values[idx])
        deltas = d_reward + gamma * d_value[1:] - d_value[:-1]
        adv_tensor = discounted_cumulative_sums(deltas, gamma * lam)
        all_advantages[idx] = adv_tensor

        ret_tensor = discounted_cumulative_sums(d_reward, gamma * lam)
        ret_tensor = np.array(ret_tensor, dtype=np.float32)
        all_returns[idx] = ret_tensor

    advantage_mean, advantage_std = (
        np.mean(all_advantages),
        np.std(all_advantages)
    )
    all_advantages = (all_advantages - advantage_mean) / advantage_std

    # t6 = time.time()
    # print("Time to compute advantages: ", t6-t5)

    observation_tensor = []
    action_tensor = []
    logprob_tensor = []
    advantage_tensor = []
    return_tensor = []
    weights_tensor = []
    for batch_element_idx in range(mini_batch_size):
        obs = observation[batch_element_idx]
        for idx in range(len(obs)):
            obs_fragment = obs[:idx+1]
            while len(obs_fragment) < num_actions:
                obs_fragment.append(0)
            observation_tensor.append(obs_fragment)
            action_tensor.append(actions[batch_element_idx][idx])
            logprob_tensor.append(logprobs[batch_element_idx][idx])
            advantage_tensor.append(all_advantages[batch_element_idx][idx])
            return_tensor.append(all_returns[batch_element_idx][idx])
            weights_tensor.append(weights[batch_element_idx])

    observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to(device)
    action_tensor = torch.tensor(action_tensor, dtype=torch.int32).to(device)
    logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to(device)
    advantage_tensor = torch.tensor(np.array(advantage_tensor), dtype=torch.float32).to(device)
    return_tensor = torch.tensor(np.array(return_tensor), dtype=torch.float32).to(device)
    weights_tensor = torch.tensor(np.array(weights_tensor), dtype=torch.float32).to(device)

    # t7 = time.time()
    # print("Time to create tensors: ", t7-t6)

    targetkl = params[3]
    actor_iterations = params[7]
    for i in range(actor_iterations):
        loss,kl = actor.module.ppo_update(
            observation_tensor,
            action_tensor,
            logprob_tensor,
            advantage_tensor,
            numComponents
        )
        if kl > 1.5*targetkl:
            print("KL Breached Limit!")
            break
    # print("Actor Loss: ", loss)

    # t8 = time.time()
    # print("Time to update actor: ", t8-t7)

    critic_iterations = params[7]
    for i in range(critic_iterations):
        c_loss = critic.module.ppo_update(
            observation_tensor,
            return_tensor,
            weights_tensor
        )

    # t9 = time.time()
    # print("Time to update critic: ", t9-t8)
    avgCost = np.mean(cost,0)
    if rank == 0:
        print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', avgCost, "\n")


    return actor, critic, NFE, allDes, allCosts, avgCost, c_loss, loss, kl
