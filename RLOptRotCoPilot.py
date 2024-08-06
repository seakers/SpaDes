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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_components = 15
        self.state_dim = self.num_components * 4  # 60 vars (x, y, z, rot) * components
        self.num_actions = 100
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=64, activation='relu')
        self.hidden_2 = layers.Dense(units=64, activation='relu')
        self.hidden_3 = layers.Dense(units=64, activation='relu')
        self.position_output_layer = layers.Dense(units=self.num_actions, activation='softmax')
        self.rotation_output_layer = layers.Dense(units=6, activation='softmax')

    def call(self, inputs, training=False):
        # inputs --> (batch, state_dim)
        x = inputs
        x = self.input_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        position_output = self.position_output_layer(x)
        rotation_output = self.rotation_output_layer(x)

        return position_output, rotation_output

    def sample_configuration(self, observations):
        input_observations = []
        for obs in observations:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.state_dim:
                input_obs.append(0)
            input_observations.append(input_obs)
        input_observations = tf.convert_to_tensor(input_observations, dtype=tf.float32)
        position_output, rotation_output = self(input_observations)

        position_log_probs = tf.math.log(position_output + 1e-10)
        position_samples = tf.random.categorical(position_log_probs, 1)
        position_action_ids = tf.squeeze(position_samples, axis=-1)
        position_batch_indices = tf.range(0, tf.shape(position_log_probs)[0], dtype=tf.int64)
        position_action_probs = tf.gather_nd(position_log_probs, tf.stack([position_batch_indices, position_action_ids], axis=-1))

        rotation_log_probs = tf.math.log(rotation_output + 1e-10)
        rotation_samples = tf.random.categorical(rotation_log_probs, 1)
        rotation_action_ids = tf.squeeze(rotation_samples, axis=-1)
        rotation_batch_indices = tf.range(0, tf.shape(rotation_log_probs)[0], dtype=tf.int64)
        rotation_action_probs = tf.gather_nd(rotation_log_probs, tf.stack([rotation_batch_indices, rotation_action_ids], axis=-1))

        return position_action_probs, position_action_ids, rotation_action_probs, rotation_action_ids, position_output, rotation_output

    def ppo_update(self, observation_tensor, position_action_tensor, position_logprob_tensor, rotation_action_tensor, rotation_logprob_tensor, advantage_tensor):
        clip_ratio = 0.1

        with tf.GradientTape() as tape:
            position_pred_probs, rotation_pred_probs = self.call(observation_tensor)
            position_pred_log_probs = tf.math.log(position_pred_probs + 1e-10)
            rotation_pred_log_probs = tf.math.log(rotation_pred_probs + 1e-10)

            position_log_probs = tf.reduce_sum(
                tf.one_hot(position_action_tensor, self.num_actions) * position_pred_log_probs, axis=-1
            )
            rotation_log_probs = tf.reduce_sum(
                tf.one_hot(rotation_action_tensor, 6) * rotation_pred_log_probs, axis=-1
            )

            loss = 0

            position_ratio = tf.exp(
                position_log_probs - position_logprob_tensor
            )
            position_min_advantage = tf.where(
                advantage_tensor > 0,
                (1 + clip_ratio) * advantage_tensor,
                (1 - clip_ratio) * advantage_tensor
            )
            position_policy_loss = -tf.reduce_mean(
                tf.minimum(tf.transpose(tf.multiply(position_ratio, tf.transpose(advantage_tensor))), position_min_advantage)
            )
            loss += position_policy_loss

            rotation_ratio = tf.exp(
                rotation_log_probs - rotation_logprob_tensor
            )
            rotation_min_advantage = tf.where(
                advantage_tensor > 0,
                (1 + clip_ratio) * advantage_tensor,
                (1 - clip_ratio) * advantage_tensor
            )
            rotation_policy_loss = -tf.reduce_mean(
                tf.minimum(tf.transpose(tf.multiply(rotation_ratio, tf.transpose(advantage_tensor))), rotation_min_advantage)
            )
            loss += rotation_policy_loss

        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.a_optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        position_kl = tf.reduce_mean(
            position_logprob_tensor - position_log_probs
        )
        position_kl = tf.reduce_sum(position_kl)

        rotation_kl = tf.reduce_mean(
            rotation_logprob_tensor - rotation_log_probs
        )
        rotation_kl = tf.reduce_sum(rotation_kl)

        return loss, position_kl, rotation_kl


class Critic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_components = 15
        self.state_dim = self.num_components * 4  # 60 vars (x, y, z, rot) * components
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

    def ppo_update(self, observation, return_buffer):
        with tf.GradientTape() as tape:
            pred_values = self.call(observation)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.trainable_variables)
        self.a_optimizer.apply_gradients(zip(critic_grads, self.trainable_variables))

        return value_loss


class RLWrapper():
    @staticmethod
    def run(components):
        epochs = 1500

        actor, critic = get_models()

        allLocs = []
        allDims = []
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
            locs, dims, cost, c_loss, loss, kl, NFE = run_epoch(actor, critic, components, NFE)
            allLocs.append(locs[0])  # Just first design of each minibatch
            allDims.append(dims[0])
            cost = np.array(cost)
            allCost.append(cost * 10)
            allCostAvg.append(np.mean(cost, 0) * 10)  # Average cost of all designs in epoch
            allC_loss.append(c_loss)
            allLoss.append(loss)
            allkl.append(kl)

        allCostFlat = []
        for costSet in allCost:
            for cost in costSet:
                allCostFlat.append(cost)
        allCostsnp = np.array(allCostFlat)
        metric = Hypervolume(ref_point=np.array([100, 1, 1, 1, 1]))
        hv = [metric.do(-point) for point in allCostsnp]
        for h in hv:
            if h > maxHV:
                maxHV = h
            allHV.append(maxHV)

        print('Finished')
        return allLocs, allDims, allCostAvg, epochs, allHV


def get_models():
    actor = Actor()
    critic = Critic()

    inputs = tf.zeros(shape=(1, 60))
    actor(inputs)
    critic(inputs)

    return actor, critic


def run_epoch(actor, critic, components, NFE):
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
        position_log_probs, position_sel_actions, rotation_log_probs, rotation_sel_actions, all_position_action_probs, all_rotation_action_probs = actor.sample_configuration(observation)

        position_log_probs = position_log_probs.numpy()
        position_sel_actions = position_sel_actions.numpy().tolist()
        rotation_log_probs = rotation_log_probs.numpy()
        rotation_sel_actions = rotation_sel_actions.numpy().tolist()

        for idx, action in enumerate(position_sel_actions):
            coords = np.linspace(-1, 1, 101)
            coord_selected = coords[action]
            actions[idx].append((action, rotation_sel_actions[idx]))
            designs[idx].append((coord_selected, rotation_sel_actions[idx]))
            logprobs[idx].append((position_log_probs[idx], rotation_log_probs[idx]))
            observation[idx].append(coord_selected)
            rewards[idx].append([0, 0, 0, 0, 0])

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
            locs = [des[i][0], des[i][0], des[i][0]]
            dims = components[i].dimensions

            if des[i][1] == 0:
                dims = [dims[0], dims[1], dims[2]]
            elif des[i][1] == 1:
                dims = [dims[0], dims[2], dims[1]]
            elif des[i][1] == 2:
                dims = [dims[1], dims[0], dims[2]]
            elif des[i][1] == 3:
                dims = [dims[1], dims[2], dims[0]]
            elif des[i][1] == 4:
                dims = [dims[2], dims[0], dims[1]]
            elif des[i][1] == 5:
                dims = [dims[2], dims[1], dims[0]]

            components[i].location = locs
            components[i].dimensions = dims
            desLocs.append(locs)
            desDims.append(dims)

        costVals = getCostComps(components)
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
        if len(val) < 60:
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
    position_action_tensor = []
    position_logprob_tensor = []
    rotation_action_tensor = []
    rotation_logprob_tensor = []
    advantage_tensor = []
    return_tensor = []
    for batch_element_idx in range(mini_batch_size):
        obs = observation[batch_element_idx]
        for idx in range(len(obs)):
            obs_fragment = obs[:idx + 1]
            while len(obs_fragment) < 60:
                obs_fragment.append(0)
            observation_tensor.append(obs_fragment)
            position_action_tensor.append(actions[batch_element_idx][idx][0])
            position_logprob_tensor.append(logprobs[batch_element_idx][idx][0])
            rotation_action_tensor.append(actions[batch_element_idx][idx][1])
            rotation_logprob_tensor.append(logprobs[batch_element_idx][idx][1])
            advantage_tensor.append(all_advantages[batch_element_idx][idx])
            return_tensor.append(all_returns[batch_element_idx][idx])

    observation_tensor = tf.convert_to_tensor(observation_tensor, dtype=tf.float32)
    position_action_tensor = tf.convert_to_tensor(position_action_tensor, dtype=tf.int32)
    position_logprob_tensor = tf.convert_to_tensor(position_logprob_tensor, dtype=tf.float32)
    rotation_action_tensor = tf.convert_to_tensor(rotation_action_tensor, dtype=tf.int32)
    rotation_logprob_tensor = tf.convert_to_tensor(rotation_logprob_tensor, dtype=tf.float32)
    advantage_tensor = tf.convert_to_tensor(advantage_tensor, dtype=tf.float32)
    return_tensor = tf.convert_to_tensor(return_tensor, dtype=tf.float32)

    targetkl = 0.01
    actor_iterations = 5
    for i in range(actor_iterations):
        loss, position_kl, rotation_kl = actor.ppo_update(
            observation_tensor,
            position_action_tensor,
            position_logprob_tensor,
            rotation_action_tensor,
            rotation_logprob_tensor,
            advantage_tensor
        )
        if position_kl > 1.5 * targetkl or rotation_kl > 1.5 * targetkl:
            print("KL Breached Limit!")
            break

    critic_iterations = 5
    for i in range(critic_iterations):
        c_loss = critic.ppo_update(
            observation_tensor,
            return_tensor
        )

    tf.print('Critic Loss: ', c_loss, '\nActor Loss: ', loss, '\nAvg Cost: ', np.mean(cost, 0), "\n")

    return allLocs, allDims, cost, c_loss, loss, position_kl, NFE
