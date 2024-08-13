import numpy as np
import keras
from keras import layers
from ConfigurationCost import *
from pymoo.indicators.hv import Hypervolume
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.signal


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class Actor(tf.keras.Model):
    def __init__(self, num_components, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.state_dim = self.num_components * 4  # 60 vars (x, y, z, rot) * components
        self.position_actions = 51 # one spot for every 4 cm
        self.rotation_actions = 6 # 6 possible right angle orientations
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=64, activation='relu')
        self.hidden_2 = layers.Dense(units=64, activation='relu')
        self.hidden_3 = layers.Dense(units=64, activation='relu')
        self.position_output_layer = layers.Dense(units=self.position_actions, activation='softmax')
        self.rotation_output_layer = layers.Dense(units=self.rotation_actions, activation='softmax') # 6 actions for each possible right angle orientation

    def call(self, inputs, rot=0, training=False): # rot is 0 for position actions, 1 for rotation actions
        # inputs --> (batch, state_dim)
        x = inputs
        x = self.input_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        if rot == 0:
            x = self.position_output_layer(x)
        else:
            x = self.rotation_output_layer(x)

        return x

    def sample_configuration(self, observations, rot):
        input_observations = []
        for obs in observations:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.state_dim:
                input_obs.append(0)
            input_observations.append(input_obs)
        input_observations = tf.convert_to_tensor(input_observations, dtype=tf.float32)
        output = self(input_observations, rot=rot)  # shape: (num components, state variables)

        log_probs = tf.math.log(output + 1e-10)
        samples = tf.random.categorical(log_probs, 1)
        action_ids = tf.squeeze(samples, axis=-1)
        batch_indices = tf.range(0, tf.shape(log_probs)[0], dtype=tf.int64)
        action_probs = tf.gather_nd(log_probs, tf.stack([batch_indices, action_ids], axis=-1))
        return action_probs, action_ids, output
    
    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):
        clip_ratio = 0.1

        with tf.GradientTape() as tape:
            position_observation_tensor = tf.boolean_mask(observation_tensor, tf.math.logical_not(tf.range(len(observation_tensor)) % 4 == 3))
            rotation_observation_tensor = tf.boolean_mask(observation_tensor, tf.range(len(observation_tensor)) % 4 == 3)

            position_action_tensor = tf.boolean_mask(action_tensor, tf.math.logical_not(tf.range(len(action_tensor)) % 4 == 3))
            rotation_action_tensor = tf.boolean_mask(action_tensor, tf.range(len(action_tensor)) % 4 == 3)

            position_pred_probs = self.call(position_observation_tensor, rot=0)
            position_pred_log_probs = tf.math.log(position_pred_probs + 1e-10)
            position_log_probs = tf.reduce_sum(
                tf.one_hot(position_action_tensor, self.position_actions) * position_pred_log_probs, axis=-1
            )

            rotation_pred_probs = self.call(rotation_observation_tensor, rot=1)
            rotation_pred_log_probs = tf.math.log(rotation_pred_probs + 1e-10)
            rotation_log_probs = tf.reduce_sum(
                tf.one_hot(rotation_action_tensor, self.rotation_actions) * rotation_pred_log_probs, axis=-1
            )

            log_probs = tf.reshape(position_log_probs, [-1, 3])
            log_probs = tf.concat([log_probs, tf.expand_dims(rotation_log_probs, axis=-1)], axis=-1)
            log_probs = tf.reshape(log_probs, [-1])

            loss = 0

            ratio = tf.exp(
                log_probs - logprob_tensor
            )
            min_advantage = tf.where(
                advantage_tensor > 0,
                (1 + clip_ratio) * advantage_tensor,
                (1 - clip_ratio) * advantage_tensor
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(tf.transpose(tf.multiply(ratio,tf.transpose(advantage_tensor))), min_advantage)
            )
            loss += policy_loss

            # # Entropy 
            # entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)
            # entr = tf.reduce_mean(entr)
            # # loss = loss - (0.1 * entr)



        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.a_optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        kl = tf.reduce_mean(
            logprob_tensor - log_probs
        )
        kl = tf.reduce_sum(kl)

        return loss,kl


class Critic(tf.keras.Model):
    def __init__(self, num_components, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.state_dim = self.num_components * 4  # 60 vars (x, y, z, rot) * components
        self.cost_values = 5
        self.a_optimizer = tf.keras.optimizers.Adam()
        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=64, activation='relu')
        self.hidden_2 = layers.Dense(units=64, activation='relu')
        self.hidden_3 = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=self.cost_values, activation='linear') # One output for each cost (I hope)

    def call(self, inputs, training=False):
        # inputs --> (batch, state_dim)
        x = inputs
        x = self.input_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.output_layer(x)

        return x

    def sample_critic(self, observations):
        input_observations = []
        for obs in observations:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.state_dim:
                input_obs.append(0)
            input_observations.append(input_obs)
        input_observations = tf.convert_to_tensor(input_observations, dtype=tf.float32)
        output = self(input_observations)  # shape: (num components, 5)
        return output

    def ppo_update(self, observaion, return_buffer):
        with tf.GradientTape() as tape:
            pred_values = self.call(observaion)
            value_loss = tf.reduce_mean((return_buffer -pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.trainable_variables)
        self.a_optimizer.apply_gradients(zip(critic_grads, self.trainable_variables))

        return value_loss


class RLWrapper():

    @staticmethod
    def run(components,maxCosts):
        epochs = 1500
        num_components = len(components)
        actor, critic = get_models(num_components)

        allLocs = []
        allDims = []
        allCostAvg = []
        allC_loss = []
        allLoss = []
        allkl = []
        NFE = 0
        allCost = []
        allMaxHV = []
        allHV = []
        maxHV = 0

        for x in range(epochs):
            print("Epoch: ", x)
            locs,dims,cost,c_loss,loss,kl,NFE = run_epoch(actor, critic, components, NFE, maxCosts)
            allLocs.append(locs[0]) # Just first design of each minibatch
            allDims.append(dims[0])
            cost = np.array(cost)
            allCost.append(cost*10)
            allCostAvg.append(np.mean(cost,0)*10) # Average cost of all designs in epoch
            allC_loss.append(c_loss)
            allLoss.append(loss)
            allkl.append(kl)

        # allCost = np.array(allCost)
        # for i in range(len(allCost[0])):
        #     plt.plot(range(NFE),allCost[:,i])
        # plt.legend(["overlapCostVal", "cmCostCalVal", "offAxisInertia", "onAxisInertia", "wireCostVal"])
        # plt.title("Negative Cost")
        # plt.xlabel("Number of Function Evaluations")
        # plt.show()
        allCostFlat = []
        for costSet in allCost:
            for cost in costSet:
                allCostFlat.append(cost)
        allCostsnp = np.array(allCostFlat)
        # maxCostList = getMaxCosts(components)

        metric = Hypervolume(ref_point=np.array([100,1,1,1,1]))
        hv = [metric.do(-point) for point in allCostsnp]
        for h in hv:
            allHV.append(h)
            if h > maxHV:
                maxHV = h
            allMaxHV.append(maxHV)

        # plt.plot(allHV)
        # plt.show()

        # allCostAvg = np.array(allCostAvg)
        # for i in range(len(allCostAvg[0])):
        #     plt.plot(allCostAvg[:,i])
        # plt.legend(["overlapCostVal", "cmCostCalVal", "offAxisInertia", "onAxisInertia", "wireCostVal"])
        # plt.title("Deep RL - Average Fitness for Each Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Negative Cost")
        # plt.show()

        # plt.subplot(1,3,1)
        # plt.plot(allC_loss)
        # plt.title("Critic Loss")
        # plt.xlabel("Epoch")

        # plt.subplot(1,3,2)
        # plt.plot(allLoss)
        # plt.title("Actor Loss")
        # plt.xlabel("Epoch")

        # plt.subplot(1,3,3)
        # plt.plot(allkl)
        # plt.title("KL Divergence")
        # plt.xlabel("Epoch")

        # plt.show()

        print('Finished')
        return allLocs,allDims,allCostAvg,epochs,allMaxHV,allHV


def get_models(num_components):
    actor = Actor(num_components=num_components)
    critic = Critic(num_components=num_components)

    inputs = tf.zeros(shape=(1, num_components*4))
    actor(inputs)
    critic(inputs)

    return actor, critic


def run_epoch(actor, critic, components, NFE, maxCosts):
    mini_batch_size = 128
    num_actions = len(components) * 4

    rewards = [[] for x in range(mini_batch_size)]
    actions = [[] for x in range(mini_batch_size)]
    logprobs = [[] for x in range(mini_batch_size)]
    designs = [[] for x in range(mini_batch_size)]

    observation = [[] for x in range(mini_batch_size)]
    critic_observations = [[] for x in range(mini_batch_size)]

    # 1. Sample actor
    for x in range(num_actions):
        rot = 0
        if x % 4 == 3:
            rot = 1
        log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, rot)
        # print(all_action_probs)

        log_probs = log_probs.numpy()
        sel_actions = sel_actions.numpy().tolist()
        for idx, action in enumerate(sel_actions):
            if rot == 0:
                coords = np.linspace(-1, 1, 51)
                coord_selected = coords[action]
                designs[idx].append(coord_selected)
                observation[idx].append(coord_selected)
            elif rot == 1:
                designs[idx].append(action)
                observation[idx].append(action)
            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            rewards[idx].append([0,0,0,0,0])


    # Post processing
    # - transform flattened design to configuration
    # - evaluate configuration
    # - record reward
    allLocs = []
    allDims = []
    desLocs = []
    desDims = []
    cost = []
    locs = []
    dims = []
    for idx, des in enumerate(designs):
        for i in range(len(components)):

            locs = [des[4*i],des[4*i+1],des[4*i+2]]

            dims = components[i].dimensions
            if des[4*i+3] == 1: # start at one because unchanged if 0
                dims = [dims[0],dims[2],dims[1]]
            elif des[4*i+3] == 2:
                dims = [dims[1],dims[0],dims[2]]
            elif des[4*i+3] == 3:
                dims = [dims[1],dims[2],dims[0]]
            elif des[4*i+3] == 4:
                dims = [dims[2],dims[0],dims[1]]
            elif des[4*i+3] == 5:
                dims = [dims[2],dims[1],dims[0]]

            components[i].location = locs
            components[i].dimensions = dims
            desLocs.append(locs)
            desDims.append(dims)

        costVals = getCostComps(components,maxCosts)
        NFE += 1
        adjustCostVals = []
        for costVal in costVals:
            adjustCostVals.append(-costVal * 0.1)
        cost.append(adjustCostVals)
        rewards[idx][-1] = cost[-1]
        allLocs.append(desLocs)
        allDims.append(desDims)

    # Sample Critic
    critic_observations = []
    for batch_element_idx in range(mini_batch_size):
        obs = observation[batch_element_idx]
        for idx in range(len(obs)):
            critic_obs = []
            critic_obs.extend(obs[:idx + 1])
            critic_observations.append(critic_obs)
    critic_values = critic.sample_critic(critic_observations)

    values = []
    val = []
    counter = 0
    for c in critic_values.numpy():
        if len(val) < num_actions:
            val.append(c)
        else:
            rewards[counter].append(val[-1])
            val.append(val[-1])
            counter += 1
            values.append(val)
            val = [c]
    val.append(val[-1])
    rewards[counter].append(val[-1])
    values.append(val)

    gamma = 0.99
    lam = 0.95
    all_advantages = [[] for x in range(mini_batch_size)]
    all_returns = [[] for x in range(mini_batch_size)]
    for idx in range(mini_batch_size):
        d_reward = np.array(rewards[idx])
        d_value = np.array(values[idx])
        deltas = d_reward[:-1] + gamma * d_value[1:] - d_value[:-1]
        adv_tensor = discounted_cumulative_sums(deltas, gamma * lam)
        all_advantages[idx] = adv_tensor

        ret_tensor = discounted_cumulative_sums(d_reward[:-1], gamma * lam)
        ret_tensor = np.array(ret_tensor, dtype=np.float32)
        all_returns[idx] = ret_tensor

    advantage_mean, advantage_std = (
        np.mean(all_advantages),
        np.std(all_advantages)
    )
    all_advantages = (all_advantages - advantage_mean) / advantage_std

    observation_tensor = []
    action_tensor = []
    logprob_tensor = []
    advantage_tensor = []
    return_tensor = []
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

    observation_tensor = tf.convert_to_tensor(observation_tensor, dtype=tf.float32)
    action_tensor = tf.convert_to_tensor(action_tensor, dtype=tf.int32)
    logprob_tensor = tf.convert_to_tensor(logprob_tensor, dtype=tf.float32)
    advantage_tensor = tf.convert_to_tensor(advantage_tensor, dtype=tf.float32)
    return_tensor = tf.convert_to_tensor(return_tensor, dtype=tf.float32)

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

    critic_iterations = 5
    for i in range(critic_iterations):
        c_loss = critic.ppo_update(
            observation_tensor,
            return_tensor
        )
    tf.print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', np.mean(cost,0), "\n")


    return allLocs, allDims, cost, c_loss, loss, kl, NFE
