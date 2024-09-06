import numpy as np
import keras
from keras import layers
from ConfigurationCost import *
import tensorflow as tf
import scipy.signal
from ConfigUtils import getOrientation
from HypervolumeUtils import HypervolumeGrid


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class Actor(tf.keras.Model):
    def __init__(self, num_components, num_panels, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.state_dim = self.num_components*4  # 60 vars (panel, x, y, rot) * components
        self.position_actions = 51 # for x and y loc on each panel
        self.rotation_actions = 24 # 24 possible right angle orientations
        self.panel_actions = 2*num_panels # one for each side of each panel
        self.a_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=128, activation='relu')
        self.hidden_2 = layers.Dense(units=128, activation='relu')
        self.hidden_3 = layers.Dense(units=128, activation='relu')
        self.position_output_layer = layers.Dense(units=self.position_actions, activation='softmax')
        self.rotation_output_layer = layers.Dense(units=self.rotation_actions, activation='softmax')
        self.panel_output_layer = layers.Dense(units=self.panel_actions, activation='softmax')

    def call(self, inputs, act=0, training=False): # act is 0 for position actions, 1 for rotation actions, 2 for panel actions
        # inputs --> (batch, state_dim)
        x = inputs
        x = self.input_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        if act == 0:
            x = self.position_output_layer(x)
        elif act == 1:
            x = self.rotation_output_layer(x)
        elif act == 2:
            x = self.panel_output_layer(x)

        return x

    def sample_configuration(self, observations, act):
        input_observations = []
        for obs in observations:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.state_dim:
                input_obs.append(0)
            input_observations.append(input_obs)
        input_observations = tf.convert_to_tensor(input_observations, dtype=tf.float32)
        output = self(input_observations, act=act)  # shape: (num components, state variables)

        log_probs = tf.math.log(output + 1e-10)
        samples = tf.random.categorical(log_probs, 1)
        action_ids = tf.squeeze(samples, axis=-1)
        batch_indices = tf.range(0, tf.shape(log_probs)[0], dtype=tf.int64)
        action_probs = tf.gather_nd(log_probs, tf.stack([batch_indices, action_ids], axis=-1))
        return action_probs, action_ids, output
    
    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):
        clip_ratio = 0.1

        with tf.GradientTape() as tape:
            panel_observation_tensor = tf.boolean_mask(observation_tensor, tf.range(len(observation_tensor)) % 4 == 0)
            position_observation_tensor = tf.boolean_mask(observation_tensor, tf.math.logical_or(tf.range(len(observation_tensor)) % 4 == 1, tf.range(len(observation_tensor)) % 4 == 2))
            rotation_observation_tensor = tf.boolean_mask(observation_tensor, tf.range(len(observation_tensor)) % 4 == 3)

            panel_action_tensor = tf.boolean_mask(action_tensor, tf.range(len(action_tensor)) % 4 == 0)
            position_action_tensor = tf.boolean_mask(action_tensor, tf.math.logical_or(tf.range(len(action_tensor)) % 4 == 1, tf.range(len(action_tensor)) % 4 == 2))
            rotation_action_tensor = tf.boolean_mask(action_tensor, tf.range(len(action_tensor)) % 4 == 3)

            panel_pred_probs = self.call(panel_observation_tensor, act=2)
            panel_pred_log_probs = tf.math.log(panel_pred_probs + 1e-10)
            panel_log_probs = tf.reduce_sum(
                tf.one_hot(panel_action_tensor, self.panel_actions) * panel_pred_log_probs, axis=-1
            )

            position_pred_probs = self.call(position_observation_tensor, act=0)
            position_pred_log_probs = tf.math.log(position_pred_probs + 1e-10)
            position_log_probs = tf.reduce_sum(
                tf.one_hot(position_action_tensor, self.position_actions) * position_pred_log_probs, axis=-1
            )

            rotation_pred_probs = self.call(rotation_observation_tensor, act=1)
            rotation_pred_log_probs = tf.math.log(rotation_pred_probs + 1e-10)
            rotation_log_probs = tf.reduce_sum(
                tf.one_hot(rotation_action_tensor, self.rotation_actions) * rotation_pred_log_probs, axis=-1
            )

            log_probs = tf.reshape(position_log_probs, [-1, 2])
            log_probs = tf.concat([tf.expand_dims(panel_log_probs, axis=-1), log_probs, tf.expand_dims(rotation_log_probs, axis=-1)], axis=-1)
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
        self.state_dim = self.num_components * 4  # 60 vars (panel, x, y, rot) * components
        self.cost_values = 6 # number of cost values
        self.a_optimizer = tf.keras.optimizers.Adam()
        self.input_layer = layers.Dense(units=self.state_dim, activation='linear')
        self.hidden_1 = layers.Dense(units=128, activation='relu')
        self.hidden_2 = layers.Dense(units=128, activation='relu')
        self.hidden_3 = layers.Dense(units=128, activation='relu')
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
    def run(components,structPanels,maxCosts):
        epochs = 10
        num_components = len(components)
        num_panels = len(structPanels)
        actor, critic = get_models(num_components, num_panels)

        NFE = 0
        allHV = []
        HVgrid = HypervolumeGrid([1,1,1,1,1,1])

        for x in range(epochs):
            print("Epoch: ", x)
            NFE, HVgrid, allHV  = run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV)

        pfSolutions = HVgrid.paretoFrontSolution
        pfCosts = HVgrid.paretoFrontPoint

        print('Finished')
        return epochs,allHV,pfSolutions,pfCosts


def get_models(num_components, num_panels):
    actor = Actor(num_components=num_components, num_panels=num_panels)
    critic = Critic(num_components=num_components)

    inputs = tf.zeros(shape=(1, num_components*4))
    actor(inputs)
    critic(inputs)

    return actor, critic


def run_epoch(actor, critic, components, structPanels, NFE, maxCosts, HVgrid, allHV):
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
        act = 0
        if x % 4 == 3:
            act = 1
        elif x % 4 == 0:
            act = 2
        log_probs, sel_actions, all_action_probs = actor.sample_configuration(observation, act)
        # print(all_action_probs)

        log_probs = log_probs.numpy()
        sel_actions = sel_actions.numpy().tolist()
        for idx, action in enumerate(sel_actions):
            if act == 0:
                coords = np.linspace(-1, 1, 51)
                coord_selected = coords[action]
                designs[idx].append(coord_selected)
                observation[idx].append(coord_selected)
            elif act == 1:
                orientation_norm = np.linspace(0, 1, 24)
                designs[idx].append(action)
                observation[idx].append(orientation_norm[action])
            elif act == 2:
                panel_norm = np.linspace(0, 1, 2*len(structPanels))
                designs[idx].append(action)
                observation[idx].append(panel_norm[action])
            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            rewards[idx].append([0,0,0,0,0,0])


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
        HVgrid.updateHV(costVals, des)
        allHV.append(HVgrid.getHV())
        adjustCostVals = []
        for costVal in costVals:
            adjustCostVals.append(-costVal)
        cost.append(adjustCostVals)
        rewards[idx][-1] = cost[-1]

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


    return NFE, HVgrid, allHV
