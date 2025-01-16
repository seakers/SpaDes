import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ConfigurationCost import *
import scipy.signal
from ConfigUtils import getOrientation
from HypervolumeUtils import HypervolumeGrid, mergeHV
import time
from TransformerArchitecture import *
from DDPsetup import setup, cleanup
import os
import datetime
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle

def run(components,structPanels,maxCosts,date_str,params):
    world_size = torch.cuda.device_count()
    print("Available GPUs:", torch.cuda.device_count())

    mp.spawn(run_ddp, args=(world_size, components, structPanels, maxCosts, date_str, params), nprocs=world_size, join=True)
    
    # Load saved data
    NFE, allDes, allCosts, avgCosts = load_results(world_size)
    HVgrid = HypervolumeGrid([1,1,1,1,1])
    allHV = []
    for i in range(len(allDes)):
        if allDes[i][0] != -1:
            HVgrid.updateHV(allCosts[i], allDes[i])
        allHV.append(HVgrid.getHV())

    return NFE, allHV, HVgrid, avgCosts
    
def load_results(world_size):
    NFE = world_size * np.load(f'NFE0.npy')
    avgCosts = []
    allDes = []
    allCosts = []
    for rank in range(world_size):
        avgCosts.append(np.load(f'avgCosts{rank}.npy'))
        allDes.append(np.load(f'allDes{rank}.npy'))
        allCosts.append(np.load(f'allCosts{rank}.npy'))

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

def run_ddp(rank, world_size, components, structPanels, maxCosts, date_str, params):

    setup(rank, world_size)

    # Training code here
    run_training(rank, world_size, components, structPanels, maxCosts, date_str, params)

    # Always cleanup even if an error occurs
    cleanup()


def run_training(rank, world_size, components, structPanels, maxCosts, date_str, params):
    # Environment setup
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['NCCL_DEBUG'] = 'INFO'

    device = torch.device(f'cuda:{rank}')
    # print("Device is: ", device)
    # print("Rank is: ", rank)
    torch.cuda.set_device(device)
    epochs = params[1]
    num_components = len(components)
    num_panels = len(structPanels)

    # Initialize DDP models
    actor, critic = get_new_models(num_components, num_panels, device, params)
    # actor = Actor(num_components, num_panels).to(device)
    # critic = Critic().to(device)
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
            actor, critic, components, structPanels, NFE, maxCosts, allDes, allCosts, device, rank, params
        )

        allCLoss.append(c_loss)
        allLoss.append(loss)
        allKL.append(kl)
        avgCosts.append(avgCost)


    torch.save(actor.state_dict(), 'actorTransformer.pth')
    torch.save(critic.state_dict(), 'criticTransformer.pth')

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

def get_new_models(num_components, num_panels, device, params):
    actor = Actor(num_components=num_components, num_panels=num_panels, device=device, params=params)
    critic = Critic(num_components=num_components, device=device, params=params)

    actor.to(device)
    critic.to(device)

    inputs = torch.zeros(size=(1,num_components*4)).to(device)
    # actor = torch.jit.trace(actor, inputs)
    # critic = torch.jit.trace(critic, inputs)
    actor(inputs)
    critic(inputs)

    return actor, critic

def get_existing_models(num_components, num_panels, device):
    actor = Actor(num_components=num_components, num_panels=num_panels, device=device)
    critic = Critic(num_components=num_components, device=device)

    actor.to(device)
    critic.to(device)

    actor.load_state_dict(torch.load('actor.pth'))
    critic.load_state_dict(torch.load('critic.pth'))

    inputs = torch.zeros(size=(1,num_components*4)).to(device)
    # actor = torch.jit.trace(actor, inputs)
    # critic = torch.jit.trace(critic, inputs)
    actor(inputs)
    critic(inputs)

    return actor, critic

def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, allDes, allCosts, device, rank, params):
    # time everything
    # t0 = time.time()
    mini_batch_size = params[0]
    num_actions = len(components) * 4

    rewards = [[] for x in range(mini_batch_size)]
    actions = [[] for x in range(mini_batch_size)]
    logprobs = [[] for x in range(mini_batch_size)]
    designs = [[] for x in range(mini_batch_size)]

    observation = [[] for x in range(mini_batch_size)]
    critic_observations = [[] for x in range(mini_batch_size)]

    # t1 = time.time()
    # print("Time to initialize: ", t1-t0)

    act_list = [x % 4 for x in range(num_actions)]

    panel_norm = np.linspace(0, 1, 2 * len(structPanels))
    coords = np.linspace(-1, 1, 51)
    orientation_norm = np.linspace(0, 1, 24)

    # Create a set of six random weights
    weightsNonNorm = np.random.rand(mini_batch_size,5)
    weights = weightsNonNorm / weightsNonNorm.sum(axis=1, keepdims=True)
    # weights = np.hstack((np.full((mini_batch_size, 1), 10), weightsNonConstraint))

    # 1. Sample actor
    for x in range(num_actions):
        act = act_list[x]
        log_probs, sel_actions, all_action_probs = actor.module.sample_configuration(observation, act)
        # print("Device: ", device)
        # print("Outside: input size", len(observation), "output_size", sel_actions.size())
        # print(all_action_probs)

        log_probs = log_probs.tolist()
        sel_actions = sel_actions.tolist()
        for idx, action in enumerate(sel_actions):
            if act == 0: # panel
                designs[idx].append(action)
                observation[idx].append(panel_norm[action])
            elif act == 1: # xloc
                coord_selected = coords[action]
                designs[idx].append(coord_selected)
                observation[idx].append(coord_selected)
            elif act == 2: # yloc
                coord_selected = coords[action]
                designs[idx].append(coord_selected)
                observation[idx].append(coord_selected)
            elif act == 3: # orientation
                designs[idx].append(action)
                observation[idx].append(orientation_norm[action])

            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            
            if act == 3:
                newOverlapCost = overlapCostSingleNP(components,designs[idx],structPanels)
                if newOverlapCost > 0.005:
                    rewards[idx].append(-100)
                else:
                    rewards[idx].append(0)
            else:
                rewards[idx].append(0)

    # t2 = time.time()
    # print("Time to sample: ", t2-t1)

    # Post processing
    # - transform flattened design to configuration
    # - evaluate configuration
    # - record reward
    
    surfNormal = np.array([0,0,1])
    cost = []
    for idx, des in enumerate(designs):
        for i in range(len(components)):

            transMat = getOrientation(int(des[4*i+3]))
            components[i].orientation = transMat
        
            panelChoice = structPanels[int(des[4*i]%len(structPanels))]
            if des[4*i] >= len(structPanels):
                surfNormal = surfNormal * -1
            
            surfLoc = np.matmul(panelChoice.orientation,np.multiply([des[4*i+1],des[4*i+2],surfNormal[2]],np.array(panelChoice.dimensions)/2))
            components[i].location = surfLoc + np.multiply(np.abs(np.matmul(transMat,np.array(components[i].dimensions)/2)),np.matmul(panelChoice.orientation,surfNormal)) + panelChoice.location

        costVals = getCostComps(components,structPanels,maxCosts)
        NFE += 1
        adjCostVals = -np.array(costVals)
        rewardCostVals = adjCostVals[1:]

        HVCosts = costVals[1:]

        if costVals[0] < 0.01:
            allDes.append(des)
            allCosts.append(HVCosts)
        else:
            allDes.append(-np.ones(len(des)))
            allCosts.append(-np.ones(len(HVCosts)))
        #     HVgrid.updateHV(HVCosts, des)
        # allHV.append(HVgrid.getHV())

        rewards[idx][-1] = rewards[idx][-1] + np.dot(weights[idx],rewardCostVals)
        cost.append(adjCostVals)

        # adjustCostVals = []
        # for costVal in costVals:
        #     adjustCostVals.append(-costVal)
        # cost.append(adjustCostVals)
        # rewards[idx][-1] = adjustCostVals

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
            advantage_tensor
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
