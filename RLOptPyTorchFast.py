import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ConfigurationCost import *
import scipy.signal
from ConfigUtils import getOrientation
from HypervolumeUtils import HypervolumeGrid
import time

import os
os.environ["TORCH_ALLOW_INPLACE"] = "0"


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    xrev = torch.flip(x, [1]) * discount
    discounted_cumulative_sums = torch.cumsum(xrev, dim=1)

    return torch.flip(discounted_cumulative_sums, [1])

class Actor(nn.Module):
    def __init__(self, num_components, num_panels):
        super(Actor, self).__init__()
        self.num_components = num_components
        self.state_dim = self.num_components * 4
        self.position_actions = 51
        self.rotation_actions = 24
        self.panel_actions = 2 * num_panels
        self.dense_dim = 128
        # self.dense_dim = 32
        
        # MLP
        self.input_layer = nn.Linear(self.state_dim, self.dense_dim)
        self.input_act = nn.ReLU()

        self.hidden_1 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_1_act = nn.ReLU()

        self.hidden_2 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_2_act = nn.ReLU()

        self.hidden_3 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_3_act = nn.ReLU()

        # Output layers for each action type
        self.position_output_layer = nn.Linear(self.dense_dim, self.position_actions)
        self.rotation_output_layer = nn.Linear(self.dense_dim, self.rotation_actions)
        self.panel_output_layer = nn.Linear(self.dense_dim, self.panel_actions)

        self.output_act = nn.Softmax(dim=-1)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs, act=0):
        x = inputs.clone()

        # Pass through MLP
        x = self.input_layer(x).clone()
        x = self.input_act(x).clone()
        x = self.hidden_1(x).clone()
        x = self.hidden_1_act(x).clone()
        x = self.hidden_2(x).clone()
        x = self.hidden_2_act(x).clone()
        x = self.hidden_3(x).clone()
        x = self.hidden_3_act(x).clone()

        # Process output based on `act` flag
        if act == 0:  # Panel prediction
            x = self.panel_output_layer(x).clone()
        elif act == 1:  # X location prediction
            x = self.position_output_layer(x).clone()
        elif act == 2:  # Y location prediction
            x = self.position_output_layer(x).clone()
        elif act == 3:  # Rotation prediction
            x = self.rotation_output_layer(x).clone()

        x = self.output_act(x).clone()

        return x.clone()

    def sample_configuration(self, observations, act):
        output = self(observations, act=act)

        log_probs = torch.log(output + 1e-10)
        samples = torch.distributions.categorical.Categorical(logits=log_probs).sample()
        action_ids = samples.squeeze()
        action_probs = log_probs.gather(1, action_ids.unsqueeze(1)).squeeze()

        return action_probs, action_ids, output

    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):
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
        panel_pred_probs = self(panel_observation_tensor, act=0)
        panel_pred_log_probs = torch.log(panel_pred_probs + 1e-10)
        panel_log_probs = torch.sum(
            F.one_hot(panel_action_tensor, num_classes=self.panel_actions) * panel_pred_log_probs, dim=-1
        )

        # Compute predicted log probabilities for x position actions
        x_position_pred_probs = self(x_position_observation_tensor, act=1)
        x_position_pred_log_probs = torch.log(x_position_pred_probs + 1e-10)
        x_position_log_probs = torch.sum(
            F.one_hot(x_position_action_tensor, num_classes=self.position_actions) * x_position_pred_log_probs, dim=-1
        )

        # Compute predicted log probabilities for y position actions
        y_position_pred_probs = self(y_position_observation_tensor, act=2)
        y_position_pred_log_probs = torch.log(y_position_pred_probs + 1e-10)
        y_position_log_probs = torch.sum(
            F.one_hot(y_position_action_tensor, num_classes=self.position_actions) * y_position_pred_log_probs, dim=-1
        )

        rotation_pred_probs = self(rotation_observation_tensor, act=3)
        rotation_pred_log_probs = torch.log(rotation_pred_probs + 1e-10)
        rotation_log_probs = torch.sum(
            F.one_hot(rotation_action_tensor, num_classes=self.rotation_actions) * rotation_pred_log_probs, dim=-1
        )

        # Reshape and concatenate log probabilities
        log_probs = torch.cat([panel_log_probs.unsqueeze(-1), x_position_log_probs.unsqueeze(-1), 
                               y_position_log_probs.unsqueeze(-1), rotation_log_probs.unsqueeze(-1)], dim=-1)
        log_probs_adj = log_probs.view(-1)

        # Calculate the loss
        ratio = torch.exp(log_probs_adj - logprob_tensor)
        min_advantage = torch.where(
            advantage_tensor > 0,
            (1 + clip_ratio) * advantage_tensor,
            (1 - clip_ratio) * advantage_tensor
        )
        policy_loss = -torch.mean(torch.min(torch.t(ratio * torch.t(advantage_tensor)), min_advantage))

        # Calculate gradients and update the policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Calculate the KL divergence
        kl = torch.mean(logprob_tensor - log_probs_adj)

        return policy_loss.item(), kl.item()


class Critic(nn.Module):
    def __init__(self, num_components):
        super(Critic, self).__init__()
        self.state_dim = 4 * num_components
        self.num_objectives = 6
        self.dense_dim = 128
        # self.dense_dim = 32

        # MLP
        self.input_layer = nn.Linear(self.state_dim, self.dense_dim)
        self.input_act = nn.ReLU()

        self.hidden_1 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_1_act = nn.ReLU()

        self.hidden_2 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_2_act = nn.ReLU()

        self.hidden_3 = nn.Linear(self.dense_dim, self.dense_dim)
        self.hidden_3_act = nn.ReLU()

        # Output layer
        self.output_layer = nn.Linear(self.dense_dim, self.num_objectives)
        self.output_act = nn.Softmax(dim=-1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def create_transformer_block(self, dim, num_heads):
        return nn.TransformerEncoderLayer(dim, num_heads)

    def forward(self, inputs):
        x = inputs.clone()

        # Pass through MLP
        x = self.input_layer(x).clone()
        x = self.input_act(x).clone()
        x = self.hidden_1(x).clone()
        x = self.hidden_1_act(x).clone()
        x = self.hidden_2(x).clone()
        x = self.hidden_2_act(x).clone()
        x = self.hidden_3(x).clone()
        x = self.hidden_3_act(x).clone()

        # Output
        x = self.output_layer(x).clone()
        x = self.output_act(x).clone()

        return x.clone()

    def sample_critic(self, observations):
        output = self(observations)
        output_last = output
        
        return output_last

    def ppo_update(self, observation, return_buffer):
        pred_values = self(observation)
        value_loss = torch.mean((return_buffer - pred_values) ** 2)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()


        return value_loss.item()

class RLWrapper():

    @staticmethod
    def run(components,structPanels,maxCosts):
        torch.autograd.set_detect_anomaly(True)
        epochs = 10
        num_components = len(components)
        num_panels = len(structPanels)
        actor, critic = get_models(num_components, num_panels)

        NFE = 0
        allHV = []
        HVgrid = HypervolumeGrid([1,1,1,1,1,1])

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
        plt.show()

        pfSolutions = HVgrid.paretoFrontSolution
        pfCosts = HVgrid.paretoFrontPoint

        print('Finished')
        return epochs,allHV,pfSolutions,pfCosts


def get_models(num_components, num_panels):
    actor = Actor(num_components=num_components, num_panels=num_panels)
    critic = Critic(num_components=num_components)

    actor = actor.to('cuda')
    critic = critic.to('cuda')
    # actor = nn.DataParallel(actor).to('cuda')
    # critic = nn.DataParallel(critic).to('cuda')

    inputs = torch.zeros(size=(1,num_components*4)).to('cuda')
    actor(inputs)
    critic(inputs)

    return actor, critic


def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV):
    mini_batch_size = 64
    num_actions = len(components) * 4

    # Preallocate data as tensors on GPU
    rewards = torch.zeros((mini_batch_size, num_actions, 6), dtype=torch.float32, device='cuda')
    actions = torch.zeros((mini_batch_size, num_actions), dtype=torch.int32, device='cuda')
    logprobs = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32, device='cuda')
    designs = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32, device='cuda')
    observation = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32, device='cuda')
    observation_full = torch.zeros((mini_batch_size, num_actions, num_actions), dtype=torch.float32, device='cuda')
    critic_values = torch.zeros((num_actions, mini_batch_size, 6), dtype=torch.float32, device='cuda')

    with torch.no_grad():  # Disable gradient computation
        for x in range(num_actions):
            act = x % 4
            log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, act)
            crit_vals = critic.sample_critic(observation)
            critic_values[x] = crit_vals

            if act == 0:  # panel
                panel_norm = torch.linspace(0, 1, 2 * len(structPanels), device='cuda')
                designs[:, x] = sel_actions
                observation[:, x] = panel_norm[sel_actions]
            elif act in [1, 2]:  # xloc or yloc
                coords = torch.linspace(-1, 1, 51, device='cuda')
                coord_selected = coords[sel_actions]
                designs[:, x] = coord_selected
                observation[:, x] = coord_selected
            elif act == 3:  # orientation
                orientation_norm = torch.linspace(0, 1, 24, device='cuda')
                designs[:, x] = sel_actions
                observation[:, x] = orientation_norm[sel_actions]

            actions[:, x] = sel_actions
            logprobs[:, x] = log_probs
            observation_full[:, x] = observation

            if act == 3:
                newOverlapCost = overlapCostSingle(components, designs[:, :x+1], structPanels)
                rewards[:, x, 0] = -newOverlapCost / maxCosts[0]

    # Cost calculations on the CPU (may need more GPU optimization)
    surfNormal = np.array([0, 0, 1], dtype=np.float32)
    cost = np.zeros((mini_batch_size, 6), dtype=np.float32)

    for idx in range(mini_batch_size):
        # Modify components' properties
        for i in range(len(components)):
            transMat = getOrientation(int(designs[idx, 4 * i + 3].item()))
            components[i].orientation = transMat
            panelChoice = structPanels[int(designs[idx, 4 * i].item()) % len(structPanels)]
            if designs[idx, 4 * i].item() >= len(structPanels):
                surfNormal = surfNormal * -1

            surfLoc = np.matmul(panelChoice.orientation, np.multiply(np.array([designs[idx, 4 * i + 1].item(), designs[idx, 4 * i + 2].item(), surfNormal[2]]), np.array(panelChoice.dimensions) / 2))
            components[i].location = surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(components[i].dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location

        costVals = getCostComps(components, structPanels, maxCosts)
        NFE += 1
        HVgrid.updateHV(costVals, designs[idx].cpu().numpy())
        allHV.append(HVgrid.getHV())
        adjustCostVals = [-costVal for costVal in costVals]
        cost[idx] = adjustCostVals
        rewards[idx, -1] = torch.tensor(cost[idx], dtype=torch.float32, device='cuda')

    # Calculate advantages
    values = critic_values.permute(1, 0, 2)
    values = torch.cat((values, values[:, -1:, :]), dim=1)
    gamma = 0.99
    lam = 0.95
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
    advantages = discounted_cumulative_sums(deltas, gamma * lam)
    returns = discounted_cumulative_sums(rewards, gamma * lam)

    advantages = (advantages - advantages.mean()) / advantages.std()

    observation_tensor = observation_full.view(-1, num_actions)
    action_tensor = actions.view(-1)
    logprob_tensor = logprobs.view(-1)
    advantage_tensor = advantages.view(-1, 6)
    return_tensor = returns.view(-1, 6)

    # Actor and Critic Updates
    targetkl = 0.01
    for i in range(5):
        loss, kl = actor.ppo_update(observation_tensor.detach(), action_tensor.detach(), logprob_tensor.detach(), advantage_tensor.detach())
        if kl > 1.5 * targetkl:
            print("KL Breached Limit!")
            break

    for i in range(5):
        c_loss = critic.ppo_update(observation_tensor.detach(), return_tensor.detach())

    print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', np.mean(cost,0), "\n")

    return NFE, HVgrid, allHV, c_loss, loss, kl


# def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV):
#     mini_batch_size = 64
#     num_actions = len(components) * 4

#     rewards = torch.zeros((mini_batch_size, num_actions, 6), dtype=torch.float32).to('cuda')
#     actions = torch.zeros((mini_batch_size, num_actions), dtype=torch.int32).to('cuda')
#     logprobs = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32).to('cuda')
#     designs = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32).to('cuda')
#     observation = torch.zeros((mini_batch_size, num_actions), dtype=torch.float32).to('cuda')
#     observation_full = torch.zeros((mini_batch_size, num_actions, num_actions), dtype=torch.float32).to('cuda')
#     critic_values = torch.zeros((num_actions, mini_batch_size, 6), dtype=torch.float32).to('cuda')


#     # Sample actor
#     for x in range(num_actions):
#         act = x % 4
#         log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, act)

#         crit_vals = critic.sample_critic(observation)
#         critic_values[x] = crit_vals

#         if act == 0:  # panel
#             panel_norm = torch.linspace(0, 1, 2 * len(structPanels)).to('cuda')
#             designs[:, x] = sel_actions
#             observation[:, x] = panel_norm[sel_actions]
#         elif act in [1, 2]:  # xloc or yloc
#             coords = torch.linspace(-1, 1, 51).to('cuda')
#             coord_selected = coords[sel_actions]
#             designs[:, x] = coord_selected
#             observation[:, x] = coord_selected
#         elif act == 3:  # orientation
#             orientation_norm = torch.linspace(0, 1, 24).to('cuda')
#             designs[:, x] = sel_actions
#             observation[:, x] = orientation_norm[sel_actions]

#         actions[:, x] = sel_actions
#         logprobs[:, x] = log_probs
#         observation_full[:, x] = observation

#         if act == 3:
#             newOverlapCost = overlapCostSingle(components, designs[:, :x+1], structPanels)
#             rewards[:, x, 0] = -newOverlapCost / maxCosts[0]

#     # Post processing
#     surfNormal = np.array([0, 0, 1], dtype=np.float32)
#     cost = np.zeros((mini_batch_size, 6), dtype=np.float32)

#     for idx in range(mini_batch_size):
#         for i in range(len(components)):
#             transMat = getOrientation(int(designs[idx, 4 * i + 3].item()))
#             components[i].orientation = transMat

#             panelChoice = structPanels[int(designs[idx, 4 * i].item()) % len(structPanels)]
#             if designs[idx, 4 * i].item() >= len(structPanels):
#                 surfNormal = surfNormal * -1

#             surfLoc = np.matmul(panelChoice.orientation, np.multiply(np.array([designs[idx, 4 * i + 1].item(), designs[idx, 4 * i + 2].item(), surfNormal[2]]), np.array(panelChoice.dimensions) / 2))
#             components[i].location = surfLoc + np.multiply(np.abs(np.matmul(transMat, np.array(components[i].dimensions) / 2)), np.matmul(panelChoice.orientation, surfNormal)) + panelChoice.location

#         costVals = getCostComps(components, structPanels, maxCosts)
#         NFE += 1
#         HVgrid.updateHV(costVals, designs[idx].cpu().numpy())
#         allHV.append(HVgrid.getHV())
#         adjustCostVals = []
#         for costVal in costVals:
#             adjustCostVals.append(-costVal)        
#         cost[idx] = adjustCostVals
#         rewards[idx, -1] = torch.tensor(cost[idx], dtype=torch.float32).to('cuda')


#     values = critic_values.permute(1, 0, 2)
#     values = torch.cat((values, values[:, -1:, :]), dim=1)

#     gamma = 0.99
#     lam = 0.95
#     deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
#     advantages = discounted_cumulative_sums(deltas, gamma * lam)
#     returns = discounted_cumulative_sums(rewards, gamma * lam)

#     advantage_mean, advantage_std = advantages.mean(), advantages.std()
#     advantages = (advantages - advantage_mean) / advantage_std

#     observation_tensor = observation_full.view(-1, num_actions)
#     action_tensor = actions.view(-1)
#     logprob_tensor = logprobs.view(-1)
#     advantage_tensor = advantages.view(-1,6)
#     return_tensor = returns.view(-1,6)

#     targetkl = 0.01
#     actor_iterations = 5
#     for i in range(actor_iterations):
#         loss, kl = actor.ppo_update(
#             observation_tensor.detach(),
#             action_tensor.detach(),
#             logprob_tensor.detach(),
#             advantage_tensor.detach()
#         )
#         if kl > 1.5 * targetkl:
#             print("KL Breached Limit!")
#             break

#     critic_iterations = 5
#     for i in range(critic_iterations):
#         c_loss = critic.ppo_update(
#             observation_tensor.detach(),
#             return_tensor.detach()
#         )

#     print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', np.mean(cost,0), "\n")

#     return NFE, HVgrid, allHV, c_loss, loss, kl

