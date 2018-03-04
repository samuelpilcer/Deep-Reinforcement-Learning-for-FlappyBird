from numpy import *
from math import pi
from random import random, randint

from alife.rl.agent import Agent
from define_model import *


class DQNCustomAgent(Agent):
    def __init__(self, obs_space, act_space, gen=1, complete_model=None):
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

        if gen == 1:
            pre_trained_model = load_model('models/model_preprocessed_images.')
            complete_model = adapt_model_to_alife(pre_trained_model, input_shape=(obs_space.shape[0], 1))

        self.model = complete_model
        # self.dqn = get_agent_from_model(self.model, act_space.shape[0], SHRUNKEN_SHAPE)
        self.generation = gen
        self.memory = 100
        self.eps = EPS_MAX
        self.log = zeros((self.memory, self.obs_space.shape[0] + 3))
        self.t = 0

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

            Returns
            -------

            A number array of length L
                (the action to take)
        """
        # Save some info to a log
        D = self.obs_space.shape[0]
        self.log[self.t, 0:D] = obs
        self.log[self.t, -1] = reward
        self.t = (self.t + 1) % len(self.log)
        if self.t % len(self.log) == 0:
            self.model.fit()

        a = self.eps_policy(obs)
        # ... and clip to within the bounds of action the space.
        a = self.get_true_action_value(a)

        # More logging ...
        self.log[self.t, D:-1] = a

        # Return
        return a

    def eps_policy(self, obs):
        D = self.obs_space.shape[0]
        a = [0, 0]
        if random() > self.eps:
            actions = self.model.predict(obs.reshape((1, D, 1))).reshape(ANGLE_SHAPE, SPEED_SHAPE)
            a = list(unravel_index(argmax(actions), actions.shape))
        else:
            a[0] = randint(0, ANGLE_SHAPE)
            a[1] = randint(0, SPEED_SHAPE)
        self.eps = max(EPS_DECAY * self.eps, EPS_MIN)
        return a

    def get_true_action_value(self, a):
        a[0] = (a[0] * ANGLE_STEP - 45) * pi / 180.0
        a[0] = clip(a[0], self.act_space.low[0], self.act_space.high[0])

        a[1] = a[1] * SPEED_STEP - 10
        a[1] = clip(a[1], self.act_space.low[1], self.act_space.high[1])
        return a

    def spawn_copy(self):
        """
            Spawn.

            Returns
            -------

            A new copy (child) of this agent, [optionally] based on this one (the parent).
        """
        b = Evolver(self.obs_space, self.act_space, self.generation + 1)

        # Make a random adjustment to the weight matrix.
        b.W = (self.W + random.randn(*self.W.shape) * 0.1) * (self.W > 0.0)
        b.w = b.w + random.randn(self.act_space.shape[0]) * 0.01
        return b

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
