import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ConfigurationCost import *
import scipy.signal
from ConfigUtils import getOrientation
from HypervolumeUtils import HypervolumeGrid
import time
from torch.amp import autocast, GradScaler
from TransformerArchitecture import CustomTransformerDecoder


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):
    def __init__(self, num_components, num_panels):
        super(Actor, self).__init__()
        self.num_components = num_components
        self.state_dim = self.num_components * 4
        self.position_actions = 201
        self.rotation_actions = 24
        self.panel_actions = 2 * num_panels
        self.nhead = 2
        self.dense_dim = 512  # Hidden dimension for the transformer
        self.scaler = GradScaler('cuda')

        # Input encoder (linear embedding)
        self.encoder = nn.Linear(2, self.dense_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.state_dim, self.dense_dim))

        # Transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.dense_dim, nhead=self.nhead, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.transformer_decoder = CustomTransformerDecoder(d_model=self.dense_dim, nhead=self.nhead, num_layers=4)

        # Output layers for each action type
        self.position_output_layer = nn.Linear(self.dense_dim, self.position_actions)
        self.rotation_output_layer = nn.Linear(self.dense_dim, self.rotation_actions)
        self.panel_output_layer = nn.Linear(self.dense_dim, self.panel_actions)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def generate_square_subsequent_mask(self, sz):
        """Generates a mask to prevent the model from looking at future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, inputs, act=0):
        with autocast(device_type='cuda', dtype=torch.float16):
            actions = torch.fmod(torch.arange(0, inputs.size()[1]), 4) + 1
            actions = actions.repeat(inputs.size()[0], 1).to(inputs.device)  # Ensure it runs on the same device (e.g., 'cuda')

            # Concatenate inputs and actions, then pass through the encoder
            x = torch.cat((inputs.unsqueeze(-1), actions.unsqueeze(-1)), dim=-1)
            x = self.encoder(x)

            # Add positional encoding
            x += self.positional_encoding[:, :x.size(1), :]

            # Create a causal mask (prevents future information from leaking)
            seq_len = x.size(1)
            batch_size = x.size(0)
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
            # mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)

            # Pass through the transformer decoder (decoder-only architecture)
            memory = x  # For a decoder-only architecture, we can feed `memory` as the encoded input itself.
            x = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)

            # Process output based on `act` flag
            if act == 0:  # Panel prediction
                x = self.panel_output_layer(x)
            elif act == 1:  # X location prediction
                x = self.position_output_layer(x)
            elif act == 2:  # Y location prediction
                x = self.position_output_layer(x)
            elif act == 3:  # Rotation prediction
                x = self.rotation_output_layer(x)

            x = F.softmax(x, dim=-1)

        return x

    def sample_configuration(self, observations, act):
        if len(observations[0]) == 0:
            observations = [[0] for x in range(len(observations))]
        input_observations = torch.tensor(observations, dtype=torch.float32).to('cuda')
        output = self(input_observations, act=act)

        output_last = output[:, -1, :]
        log_probs = torch.log(output_last + 1e-10)

        samples = torch.distributions.categorical.Categorical(logits=log_probs).sample()
        action_ids = samples.squeeze()
        action_probs = log_probs.gather(1, action_ids.unsqueeze(1)).squeeze()

        return action_probs, action_ids, output_last

    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):

        self.optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):

            clip_ratio = 0.1
            
            # Separate observation tensor for panel, position, and rotation
            panel_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 0]
            x_position_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 1]
            y_position_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 2]
            rotation_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 3]

            # Separate action tensor for panel, position, and rotation
            panel_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 0].long()
            x_position_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 1].long()
            y_position_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 2].long()
            rotation_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 3].long()

            # Compute predicted log probabilities for panel, position, and rotation actions
            panel_pred_probs = self(panel_observation_tensor, act=0)[:,-1,:]
            panel_pred_log_probs = torch.log(panel_pred_probs + 1e-10)
            panel_log_probs = torch.sum(
                F.one_hot(panel_action_tensor, num_classes=self.panel_actions) * panel_pred_log_probs, dim=-1
            )

            # Compute predicted log probabilities for x position actions
            x_position_pred_probs = self(x_position_observation_tensor, act=1)[:,-1,:]
            x_position_pred_log_probs = torch.log(x_position_pred_probs + 1e-10)
            x_position_log_probs = torch.sum(
                F.one_hot(x_position_action_tensor, num_classes=self.position_actions) * x_position_pred_log_probs, dim=-1
            )

            # Compute predicted log probabilities for y position actions
            y_position_pred_probs = self(y_position_observation_tensor, act=2)[:,-1,:]
            y_position_pred_log_probs = torch.log(y_position_pred_probs + 1e-10)
            y_position_log_probs = torch.sum(
                F.one_hot(y_position_action_tensor, num_classes=self.position_actions) * y_position_pred_log_probs, dim=-1
            )

            rotation_pred_probs = self(rotation_observation_tensor, act=3)[:,-1,:]
            rotation_pred_log_probs = torch.log(rotation_pred_probs + 1e-10)
            rotation_log_probs = torch.sum(
                F.one_hot(rotation_action_tensor, num_classes=self.rotation_actions) * rotation_pred_log_probs, dim=-1
            )

            # Reshape and concatenate log probabilities
            log_probs = torch.cat([panel_log_probs.unsqueeze(-1), x_position_log_probs.unsqueeze(-1), 
                                y_position_log_probs.unsqueeze(-1), rotation_log_probs.unsqueeze(-1)], dim=-1)
            log_probs = log_probs.view(-1)

            # Calculate the loss
            ratio = torch.exp(log_probs - logprob_tensor)
            min_advantage = torch.where(
                advantage_tensor > 0,
                (1 + clip_ratio) * advantage_tensor,
                (1 - clip_ratio) * advantage_tensor
            )
            policy_loss = -torch.mean(torch.min(torch.t(ratio * torch.t(advantage_tensor)), min_advantage))

        # Calculate gradients and update the policy
        # self.optimizer.zero_grad()
        # policy_loss.backward()
        # self.optimizer.step()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        # Calculate the KL divergence
        kl = torch.mean(logprob_tensor - log_probs)

        return policy_loss.item(), kl.item()

class Critic(nn.Module):
    def __init__(self, num_components):
        super(Critic, self).__init__()
        self.num_components = num_components
        self.state_dim = self.num_components * 4
        self.dense_dim = 512  # Hidden dimension for the transformer
        self.num_objectives = 5  # Number of value predictions (e.g., one for each objective)
        self.nhead = 2
        self.scaler = GradScaler('cuda')

        # Input encoder (linear embedding)
        self.encoder = nn.Linear(2, self.dense_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.state_dim, self.dense_dim))

        # Transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.dense_dim, nhead=8, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.transformer_decoder = CustomTransformerDecoder(d_model=self.dense_dim, nhead=self.nhead, num_layers=4)

        # Output layer for value estimation
        self.value_output_layer = nn.Linear(self.dense_dim, self.num_objectives)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def generate_square_subsequent_mask(self, sz):
        """Generates a mask to prevent the model from looking at future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, inputs):
        with autocast(device_type='cuda', dtype=torch.float16):

            actions = torch.fmod(torch.arange(0, inputs.size()[1]), 4) + 1
            actions = actions.repeat(inputs.size()[0], 1).to(inputs.device)  # Ensure it runs on the same device (e.g., 'cuda')

            # Concatenate inputs and actions, then pass through the encoder
            x = torch.cat((inputs.unsqueeze(-1), actions.unsqueeze(-1)), dim=-1)
            x = self.encoder(x)

            # Add positional encoding
            x += self.positional_encoding[:, :x.size(1), :]

            # Create a causal mask (prevents future information from leaking)
            seq_len = x.size(1)
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

            # Pass through the transformer decoder (decoder-only architecture)
            memory = x  # For a decoder-only architecture, we can feed `memory` as the encoded input itself.
            x = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)

            # Pass through the value output layer (single output for value prediction)
            x = self.value_output_layer(x)

            # Return only the last value prediction, which is typically the value of the current state
            x = x[:, -1, :]

        return x

    def sample_critic(self, observations):
        input_observations = torch.tensor(observations, dtype=torch.float32).to('cuda')
        output = self(input_observations)
        
        return output

    def ppo_update(self, observation, return_buffer, weights):
        self.optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            pred_values = self(observation)
            pred_reward = torch.sum(-pred_values * weights, dim=-1)
            value_loss = torch.mean((return_buffer - pred_reward) ** 2)

        # Backpropagation
        # self.optimizer.zero_grad()
        # value_loss.backward()
        # self.optimizer.step()
        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        return value_loss.item()


class RLWrapper():

    @staticmethod
    def run(components,structPanels,maxCosts):

        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        epochs = 10
        num_components = len(components)
        num_panels = len(structPanels)
        actor, critic = get_new_models(num_components, num_panels)
        # actor, critic = get_existing_models(num_components, num_panels)

        NFE = 0
        allHV = []
        HVgrid = HypervolumeGrid([1,1,1,1,1])

        allCLoss = []
        allLoss = []
        allKL = []

        for x in range(epochs):
            print("Epoch: ", x)
            NFE, HVgrid, allHV, c_loss, loss, kl  = run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV)
            allCLoss.append(c_loss)
            allLoss.append(loss)
            allKL.append(kl)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        # Subplot for C Loss
        plt.subplot(1, 3, 1)
        plt.plot(range(epochs), allCLoss, label='C Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss vs Epochs')
        plt.grid()
        plt.legend()

        # Subplot for Loss
        plt.subplot(1, 3, 2)
        plt.plot(range(epochs), allLoss, label='Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Actor Loss')
        plt.title('Actor Loss vs Epochs')
        plt.grid()
        plt.legend()

        # Subplot for KL
        plt.subplot(1, 3, 3)
        plt.plot(range(epochs), allKL, label='KL Divergence', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence vs Epochs')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig('ResultGraphs/11-06-24/RLTraining.png')

        pfSolutions = HVgrid.paretoFrontSolution
        pfCosts = HVgrid.paretoFrontPoint

        print('Finished')
        return epochs,allHV,pfSolutions,pfCosts


def get_new_models(num_components, num_panels):
    actor = Actor(num_components=num_components, num_panels=num_panels).to('cuda')
    critic = Critic(num_components=num_components).to('cuda')

    inputs = torch.zeros(size=(1,num_components*4)).to('cuda')
    # actor = torch.jit.trace(actor, inputs)
    # critic = torch.jit.trace(critic, inputs)
    actor(inputs)
    critic(inputs)

    return actor, critic

def get_existing_models(num_components, num_panels):
    actor = Actor(num_components=num_components, num_panels=num_panels).to('cuda')
    critic = Critic(num_components=num_components).to('cuda')

    actor.load_state_dict(torch.load('actor.pth'))
    critic.load_state_dict(torch.load('critic.pth'))

    inputs = torch.zeros(size=(1,num_components*4)).to('cuda')
    # actor = torch.jit.trace(actor, inputs)
    # critic = torch.jit.trace(critic, inputs)
    actor(inputs)
    critic(inputs)

    return actor, critic

def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV):
    # time everything
    # t0 = time.time()
    mini_batch_size = 8
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
    coords = np.linspace(-1, 1, 201)
    orientation_norm = np.linspace(0, 1, 24)

    # Create a set of six random weights
    weightsNonNorm = np.random.rand(mini_batch_size,5)
    weights = weightsNonNorm / weightsNonNorm.sum(axis=1, keepdims=True)
    # weights = np.hstack((np.full((mini_batch_size, 1), 10), weightsNonConstraint))

    # 1. Sample actor
    for x in range(num_actions):
        act = act_list[x]
        log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, act)
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
            HVgrid.updateHV(HVCosts, des)
        allHV.append(HVgrid.getHV())

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
        crit_vals = critic.sample_critic(critic_observations)
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
    

    gamma = 0.99
    lam = 0.95
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

    observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to('cuda')
    action_tensor = torch.tensor(action_tensor, dtype=torch.int32).to('cuda')
    logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to('cuda')
    advantage_tensor = torch.tensor(np.array(advantage_tensor), dtype=torch.float32).to('cuda')
    return_tensor = torch.tensor(np.array(return_tensor), dtype=torch.float32).to('cuda')
    weights_tensor = torch.tensor(np.array(weights_tensor), dtype=torch.float32).to('cuda')

    # t7 = time.time()
    # print("Time to create tensors: ", t7-t6)

    targetkl = 0.01
    actor_iterations = 5
    for i in range(actor_iterations):
        loss,kl = actor.ppo_update(
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

    critic_iterations = 5
    for i in range(critic_iterations):
        c_loss = critic.ppo_update(
            observation_tensor,
            return_tensor,
            weights_tensor
        )

    # t9 = time.time()
    # print("Time to update critic: ", t9-t8)
    avgCost = np.mean(cost,0)
    print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', avgCost, "\n")


    return actor, critic, NFE, HVgrid, allHV, avgCost, c_loss, loss, kl


# def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV):
#     # time everything
#     # t0 = time.time()
#     mini_batch_size = 1024
#     num_actions = len(components) * 4

#     rewards = [[] for x in range(mini_batch_size)]
#     actions = [[] for x in range(mini_batch_size)]
#     logprobs = [[] for x in range(mini_batch_size)]
#     designs = [[] for x in range(mini_batch_size)]

#     observation = [[] for x in range(mini_batch_size)]
#     critic_observations = [[] for x in range(mini_batch_size)]

#     # t1 = time.time()
#     # print("Time to initialize: ", t1-t0)


#     # 1. Sample actor
#     for x in range(num_actions):
#         act = 0
#         if x % 4 == 1:
#             act = 1
#         elif x % 4 == 2:
#             act = 2
#         elif x % 4 == 3:
#             act = 3
#         log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, act)
#         # print(all_action_probs)

#         log_probs = log_probs.tolist()
#         sel_actions = sel_actions.tolist()
#         for idx, action in enumerate(sel_actions):
#             if act == 0: # panel
#                 panel_norm = np.linspace(0, 1, 2*len(structPanels))
#                 designs[idx].append(action)
#                 observation[idx].append(panel_norm[action])
#             elif act == 1: # xloc
#                 coords = np.linspace(-1, 1, 51)
#                 coord_selected = coords[action]
#                 designs[idx].append(coord_selected)
#                 observation[idx].append(coord_selected)
#             elif act == 2: # yloc
#                 coords = np.linspace(-1, 1, 51)
#                 coord_selected = coords[action]
#                 designs[idx].append(coord_selected)
#                 observation[idx].append(coord_selected)
#             elif act == 3: # orientation
#                 orientation_norm = np.linspace(0, 1, 24)
#                 designs[idx].append(action)
#                 observation[idx].append(orientation_norm[action])

#             actions[idx].append(action)
#             logprobs[idx].append(log_probs[idx])
#             rewards[idx].append([0,0,0,0,0,0])

#     # t2 = time.time()
#     # print("Time to sample: ", t2-t1)

#     # Post processing
#     # - transform flattened design to configuration
#     # - evaluate configuration
#     # - record reward
#     surfNormal = np.array([0,0,1])
#     cost = []
#     for idx, des in enumerate(designs):
#         for i in range(len(components)):

#             transMat = getOrientation(int(des[4*i+3]))
#             components[i].orientation = transMat
        
#             panelChoice = structPanels[int(des[4*i]%len(structPanels))]
#             if des[4*i] >= len(structPanels):
#                 surfNormal = surfNormal * -1
            
#             surfLoc = np.matmul(panelChoice.orientation,np.multiply([des[4*i+1],des[4*i+2],surfNormal[2]],np.array(panelChoice.dimensions)/2))
#             components[i].location = surfLoc + np.multiply(np.abs(np.matmul(transMat,np.array(components[i].dimensions)/2)),np.matmul(panelChoice.orientation,surfNormal)) + panelChoice.location

#         costVals = getCostComps(components,structPanels,maxCosts)
#         NFE += 1
#         HVgrid.updateHV(costVals, des)
#         allHV.append(HVgrid.getHV())
#         adjustCostVals = []
#         for costVal in costVals:
#             adjustCostVals.append(-costVal)
#         cost.append(adjustCostVals)
#         rewards[idx][-1] = cost[-1]

#     # t3 = time.time()
#     # print("Time to evaluate: ", t3-t2)

#     # Sample Critic
#     critic_values = []
#     for action_idx in range(num_actions):
#         critic_observations = []
#         for idx in range(mini_batch_size):
#             obs = observation[idx]
#             critic_obs = []
#             critic_obs.extend(obs[:action_idx + 1])
#             critic_observations.append(critic_obs)
#         crit_vals = critic.sample_critic(critic_observations)
#         crit_vals = crit_vals.squeeze().tolist()
#         critic_values.append(crit_vals)

#     # t4 = time.time()
#     # print("Time to sample critic: ", t4-t3)
    
#     values = [[] for x in range(mini_batch_size)]
#     for act_idx, act_vals in enumerate(critic_values):
#         for batch_idx, val in enumerate(act_vals):
#             values[batch_idx].append(val)

#     for idx in range(mini_batch_size):
#         values[idx].append(values[idx][-1])

#     # t5 = time.time()
#     # print("Time to process values: ", t5-t4)
    

#     gamma = 0.99
#     lam = 0.95
#     all_advantages = [[] for x in range(mini_batch_size)]
#     all_returns = [[] for x in range(mini_batch_size)]
#     for idx in range(mini_batch_size):
#         d_reward = np.array(rewards[idx])
#         d_value = np.array(values[idx])
#         deltas = d_reward + gamma * d_value[1:] - d_value[:-1]
#         adv_tensor = discounted_cumulative_sums(deltas, gamma * lam)
#         all_advantages[idx] = adv_tensor

#         ret_tensor = discounted_cumulative_sums(d_reward, gamma * lam)
#         ret_tensor = np.array(ret_tensor, dtype=np.float32)
#         all_returns[idx] = ret_tensor

#     advantage_mean, advantage_std = (
#         np.mean(all_advantages),
#         np.std(all_advantages)
#     )
#     all_advantages = (all_advantages - advantage_mean) / advantage_std

#     # t6 = time.time()
#     # print("Time to compute advantages: ", t6-t5)

#     observation_tensor = []
#     action_tensor = []
#     logprob_tensor = []
#     advantage_tensor = []
#     return_tensor = []
#     for batch_element_idx in range(mini_batch_size):
#         obs = observation[batch_element_idx]
#         for idx in range(len(obs)):
#             obs_fragment = obs[:idx+1]
#             while len(obs_fragment) < num_actions:
#                 obs_fragment.append(0)
#             observation_tensor.append(obs_fragment)
#             action_tensor.append(actions[batch_element_idx][idx])
#             logprob_tensor.append(logprobs[batch_element_idx][idx])
#             advantage_tensor.append(all_advantages[batch_element_idx][idx])
#             return_tensor.append(all_returns[batch_element_idx][idx])

#     observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to('cuda')
#     action_tensor = torch.tensor(action_tensor, dtype=torch.int32).to('cuda')
#     logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to('cuda')
#     advantage_tensor = torch.tensor(np.array(advantage_tensor), dtype=torch.float32).to('cuda')
#     return_tensor = torch.tensor(np.array(return_tensor), dtype=torch.float32).to('cuda')

#     # t7 = time.time()
#     # print("Time to create tensors: ", t7-t6)

#     targetkl = 0.01
#     actor_iterations = 5
#     for i in range(actor_iterations):
#         loss,kl = actor.ppo_update(
#             observation_tensor,
#             action_tensor,
#             logprob_tensor,
#             advantage_tensor
#         )
#         if kl > 1.5*targetkl:
#             print("KL Breached Limit!")
#             break
#     # print("Actor Loss: ", loss)

#     # t8 = time.time()
#     # print("Time to update actor: ", t8-t7)

#     critic_iterations = 5
#     for i in range(critic_iterations):
#         c_loss = critic.ppo_update(
#             observation_tensor,
#             return_tensor
#         )

#     # t9 = time.time()
#     # print("Time to update critic: ", t9-t8)
    
#     print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', np.mean(cost,0), "\n")


#     return NFE, HVgrid, allHV, c_loss, loss, kl
