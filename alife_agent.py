import random
from math import pi

import numpy as np

from alife.rl.agent import Agent
from define_model import *


class DQNCustomAgent(Agent):
    def __init__(self, obs_space, act_space, gen=1, weights=None, log=None, log_complete=None):
        """
            Init.


            Parameters
            ----------

            obs_space : BugSpace
                observation space
            act_space : BugSpace
                action space
            gen : int
                current generation

        """
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_shape = self.obs_space.shape[0]

        pre_trained_model = load_model('models/model_preprocessed_images.')
        complete_model = adapt_model_to_alife(pre_trained_model, input_shape=(obs_space.shape[0], 1))
        self.model = complete_model
        self.model.compile(Adam(lr=LEARNING_RATE), loss='mean_squared_error', metrics=['mae'])
        self.q_mask = np.ones((1, 1, ANGLE_ACTION_POSSIBILITIES, SPEED_ACTION_POSSIBILITIES))*np.nan

        self.generation = gen
        self.eps = EPS_MAX

        self.log = np.zeros((MEMORY, self.obs_shape + 3))
        self.log_complete = False

        self.t = 0
        self.to_fit = False

        if gen > 1:
            self.model.set_weights(weights)
            self.log = log
            self.log_complete = log_complete

    def __str__(self):
        ''' Return a string representation (e.g., a label) for this agent '''
        return "DQN Agent: Gen %d" % self.generation

    def act(self, obs, reward, done=False):
        """
            Act.

            Parameters
            ----------

            obs : numpy array of length D
                the state at the current time
            reward : float
                the reward signal at the current time
            done : boolean
                is current state a final state

            Returns
            -------

            A number array of length L
                (the action to take)
        """
        # Save some info to a log
        self.log_state(obs, reward)
        if self.to_fit:
            if self.log_complete:
                self.fit_model(idx_max=MEMORY - 1)
            else:
                self.fit_model(idx_max=self.t - 1)
            self.to_fit = False

        # Choose the best action to do with an Epsilon Greedy Policy
        a = self.eps_policy(obs)
        self.log_action(a)

        # Get the true value of the action (change of ange and speed)
        a = self.idx_to_action_value(a)

        # Save the action chosen
        if self.t == MEMORY - 1:
            self.log_complete = True
        if self.t % FIT_INTERVAL == 0 and (self.t > 0 or self.log_complete):
            self.to_fit = True
        self.t = (self.t + 1) % MEMORY

        return a

    def log_state(self, obs, reward):
        self.log[self.t, :self.obs_shape] = obs
        self.log[self.t, -1] = reward
        return 0

    def log_action(self, a):
        self.log[self.t, self.obs_shape:-1] = a
        return 0

    def eps_policy(self, obs):
        a = [0, 0]
        if random.random() > self.eps:
            actions = self.model.predict(obs.reshape((1, self.obs_shape, 1))). \
                reshape(ANGLE_ACTION_POSSIBILITIES, SPEED_ACTION_POSSIBILITIES)
            a = list(np.unravel_index(np.argmax(actions), actions.shape))
        else:
            a[0] = random.randint(0, ANGLE_ACTION_POSSIBILITIES - 1)
            a[1] = random.randint(0, SPEED_ACTION_POSSIBILITIES - 1)
        return a

    def idx_to_action_value(self, a):
        a[0] = (a[0] * ANGLE_STEP - 45) * pi / 180.0
        a[0] = np.clip(a[0], self.act_space.low[0], self.act_space.high[0])

        a[1] = a[1] * SPEED_STEP - 10
        a[1] = np.clip(a[1], self.act_space.low[1], self.act_space.high[1])
        return a

    def get_minibatch(self, idx_max):
        idx_list = np.random.randint(idx_max, size=BATCH_SIZE)
        batch = []
        for idx in idx_list:
            state = np.array(self.log[idx, :self.obs_shape]).reshape(1, self.obs_shape, 1)
            action = self.log[idx, self.obs_shape:-1]
            reward = self.log[idx, -1]
            if idx == idx_max:
                next_state = np.array(self.log[0, :self.obs_shape]).reshape(1, self.obs_shape, 1)
            else:
                next_state = np.array(self.log[idx + 1, :self.obs_shape]).reshape(1, self.obs_shape, 1)
            batch.append((state, action, reward, next_state))
        return batch

    def fit_model(self, idx_max):
        minibatch = self.get_minibatch(idx_max)
        state_train, target_f_train = np.zeros((BATCH_SIZE, self.obs_shape, 1)), \
                                      np.zeros((BATCH_SIZE, 1, ANGLE_ACTION_POSSIBILITIES, SPEED_ACTION_POSSIBILITIES))
        i = 0
        for (state, action, reward, next_state) in minibatch:
            self.q_mask[:, :, int(action[0]), int(action[1])] = 1
            target = reward + GAMMA * np.amax(self.model.predict(next_state))
            target_f = np.multiply(self.model.predict(state), self.q_mask)
            target_f[:, :, int(action[0]), int(action[1])] = target
            state_train[i] = state
            target_f_train[i] = target_f
            i += 1
        self.model.fit(state_train, target_f_train, epochs=1, verbose=1)
        if self.eps > EPS_MIN:
            self.eps *= EPS_DECAY

    def spawn_copy(self):
        """
            Spawn.

            Returns
            -------

            A new copy (child) of this agent, [optionally] based on this one (the parent).
        """
        new_agent = DQNCustomAgent(self.obs_space, self.act_space, self.generation + 1, self.model.get_weights(),
                                   log=self.log, log_complete=self.log_complete)

        return new_agent

    def save(self, bin_path, log_path, obj_ID):
        """
            Save a representation of this agent.

            Here we save a .csv of the state/action/reward signals.
            (such that it could be loaded later by pandas for visualization).
        """
        header = [("X%d" % j) for j in range(self.log.shape[1])]
        header[-1] = "reward"
        fname = log_path + ("/%d-%s-G%d.log" % (obj_ID, self.__class__.__name__, self.generation))
        savetxt(fname, self.log[0:self.t, :], fmt='%4.3f', delimiter=',', header=','.join(header), comments='')
        print("Saved log to '%s'." % fname)
