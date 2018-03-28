import gym
import gym_ple
import numpy as np

from define_model import load_model, get_agent_from_model
from constants import *
from post_process import test_agent

model = load_model(filename='models/model_preprocessed_images.')
print(model.summary())
env = gym.make(ENV_NAME)
np.random.seed(SEED)
env.seed(SEED)
nb_actions = env.action_space.n
obs = env.reset()

dqn = get_agent_from_model(model=model, nb_actions=nb_actions, input_shape=SHRUNKEN_SHAPE)

test_agent(env=env, agent=dqn)
