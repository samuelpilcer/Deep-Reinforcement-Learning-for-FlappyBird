from __future__ import division

import gym
import gym_ple
import numpy as np
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


from define_model import *
from constants import *
import post_process


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(SEED)
env.seed(SEED)
nb_actions = env.action_space.n
obs = env.reset()


# Define the Neural Net model and the agent
# model = get_cnn_model(input_shape=(WINDOW_LENGTH,) + SHRUNKEN_SHAPE, nb_actions=nb_actions)
model = load_model(filename='models/model_preprocessed_images.')
dqn = get_agent_from_model(model, nb_actions, SHRUNKEN_SHAPE)

callbacks = [ModelIntervalCheckpoint(CHECKPOINT_WEIGHTS_FILENAME, interval=100000)]
callbacks += [FileLogger(LOG_FILENAME, interval=100)]

history = dqn.fit(env, callbacks=callbacks, nb_steps=1500000, log_interval=10000, verbose=2)


post_process.plot_reward(history)
post_process.plot_steps(history)
post_process.test_agent(env=env, agent=dqn, nb_episodes=10)
post_process.save_model(model=model, model_json_filename=MODEL_JSON_FILENAME,
                        model_weights_filename=MODEL_WEIGHTS_FILENAME)
