import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from ConfigurationCost import *
import scipy.signal
from pymoo.indicators.hv import Hypervolume


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

import tensorflow as tf


# @keras.saving.register_keras_serializable(package='ConfigRL', name='ConfigRL')
class Actor(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_components = 15
        self.state_dim = self.num_components * 3  # 45 vars
        self.num_actions = 100
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=64, activation='relu')
        self.hidden_2 = layers.Dense(units=64, activation='relu')
        self.hidden_3 = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=self.num_actions, activation='softmax')



    def call(self, inputs, training=False):
        # inputs --> (batch, state_dim)
        x = inputs
        x = self.input_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.output_layer(x)

        return x
    

    def sample_configuration(self, observations):
        input_observations = []
        for obs in observations:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.state_dim:
                input_obs.append(0)
            input_observations.append(input_obs)
        input_observations = tf.convert_to_tensor(input_observations, dtype=tf.float32)
        output = self(input_observations)  # shape: (num components, state variables)
        

        log_probs = tf.math.log(output + 1e-10)
        samples = tf.random.categorical(log_probs, 1)
        action_ids = tf.squeeze(samples, axis=-1)
        batch_indices = tf.range(0, tf.shape(log_probs)[0], dtype=tf.int64)
        action_probs = tf.gather_nd(log_probs, tf.stack([batch_indices, action_ids], axis=-1))
        return action_probs, action_ids, output
    

    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):
        clip_ratio = 0.1


        with tf.GradientTape() as tape:
            pred_probs = self.call(observation_tensor)  # (135, 100)
            pred_log_probs = tf.math.log(pred_probs + 1e-10)
            log_probs = tf.reduce_sum(
                tf.one_hot(action_tensor, self.num_actions) * pred_log_probs, axis=-1
            )

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

            # Entropy 
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)
            entr = tf.reduce_mean(entr)
            # loss = loss - (0.1 * entr)



        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.a_optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        kl = tf.reduce_mean(
            logprob_tensor - log_probs
        )
        kl = tf.reduce_sum(kl)

        return loss,kl

    
class Critic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_components = 15
        self.state_dim = self.num_components * 3  # 45 vars
        self.a_optimizer = tf.keras.optimizers.Adam()
        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=64, activation='relu')
        self.hidden_2 = layers.Dense(units=64, activation='relu')
        self.hidden_3 = layers.Dense(units=64, activation='relu')
        self.output_layer = layers.Dense(units=5, activation='linear') # One output for each cost (I hope)



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
    def run(components):
        epochs = 10

        actor, critic = get_models()

        allDesigns = []
        allCostAvg = []
        allC_loss = []
        allLoss = []
        allkl = []
        NFE = 0
        allCost = []
        allHV = []
        maxHV = 0

        for x in range(epochs):
            print("Epoch: ", x)
            design,cost,c_loss,loss,kl,NFE = run_epoch(actor, critic, components,NFE)
            allDesigns.append(design[0]) # Just first design of each minibatch
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
        maxCostList = getMaxCosts(components)

        metric = Hypervolume(ref_point=np.array(maxCostList))
        hv = [metric.do(-point) for point in allCostsnp]
        for h in hv:
            if h > maxHV:
                maxHV = h
            allHV.append(maxHV)

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
        return allDesigns,allCostAvg,epochs,allHV


def get_models():
    actor = Actor()
    critic = Critic()

    inputs = tf.zeros(shape=(1, 45))
    actor(inputs)
    critic(inputs)


    return actor, critic


def run_epoch(actor, critic, components,NFE):
    mini_batch_size = 1
    num_actions = len(components)*3

    rewards = [[] for x in range(mini_batch_size)]
    actions = [[] for x in range(mini_batch_size)]
    logprobs = [[] for x in range(mini_batch_size)]
    designs = [[] for x in range(mini_batch_size)]

    observation = [[] for x in range(mini_batch_size)]
    critic_observations = [[] for x in range(mini_batch_size)]


    # 1. Sample actor
    for x in range(num_actions):
        log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation)
        # print(all_action_probs)

        log_probs = log_probs.numpy()
        sel_actions = sel_actions.numpy().tolist()
        for idx, action in enumerate(sel_actions):
            coords = np.linspace(-1, 1, 101)
            coord_selected = coords[action]
            actions[idx].append(action)
            designs[idx].append(coord_selected)
            logprobs[idx].append(log_probs[idx])
            observation[idx].append(coord_selected)
            rewards[idx].append([0,0,0,0,0])

    # Post processing
    # - transform flattened design to configuration
    # - evaluate configuration
    # - record reward
    # ORGANIC FREE RANGE CODE
    locations = []
    configs = []
    cost = []
    for idx, des in enumerate(designs):
        for i in range(len(components)):
            locs = [des[3*i],des[3*i+1],des[3*i+2]]
            components[i].location = locs
            locations.append(locs)
        costVals = getCostComps(components)
        NFE+=1
        adjustCostVals = []
        for costVal in costVals:
            adjustCostVals.append(-costVal*0.1)
        cost.append(adjustCostVals)
        rewards[idx][-1] = cost[-1]
        configs.append(locations)
    

    
    # print("Cost: "cost[0])

    # Sample Critic
    critic_observations = []
    for batch_element_idx in range(mini_batch_size):
        obs = observation[batch_element_idx]
        for idx in range(len(obs)):
            critic_obs = []
            critic_obs.extend(obs[:idx+1])
            critic_observations.append(critic_obs)
    critic_values = critic.sample_critic(critic_observations)
    # critic_values = tf.squeeze(critic_values, axis=-1)

    values = []
    val = []
    counter = 0
    for c in critic_values.numpy():
        if len(val) < 45:
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
            while len(obs_fragment) < 45:
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

    # print(observation_tensor)
    # print(action_tensor)
    # print(logprob_tensor)
    # print(advantage_tensor)
    # print(return_tensor)

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


    return configs,cost,c_loss,loss,kl,NFE


















