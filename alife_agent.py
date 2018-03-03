from alife.alife.rl.agent import Agent
from numpy import *
from define_model import *


class DQNCustomAgent(Agent):

    def __init__(self, obs_space, act_space, gen=1):
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
        model = load_model('models/model_preprocessed_images.')
        complete_model = adapt_model_to_alife(model, input_shape=obs_space.shape)

    def __str__(self):
        ''' Return a string representation (e.g., a label) for this agent '''
        return ("Evolver: Gen %d" % (self.generation))

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

        # No learning, just a simple linear reflex,
        a = dot(obs, self.W) + self.w
        # ... and clip to within the bounds of action the space.
        a[0] = clip(a[0], self.act_space.low[0], self.act_space.high[0])
        a[1] = clip(a[1], self.act_space.low[1], self.act_space.high[1])

        # More logging ...
        self.log[self.t, D:-1] = a

        # Return
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


