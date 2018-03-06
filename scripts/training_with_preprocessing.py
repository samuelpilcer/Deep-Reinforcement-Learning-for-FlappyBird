from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import numpy as np
import gym_ple

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import matplotlib.pyplot as plt
from cv2 import cv2

from keras import backend as K
import multiprocessing
nb_cores=multiprocessing.cpu_count()
K.set_session(K.tf.Session(config=K.tf.ConfigProto(device_count = {'CPU': nb_cores} )))

print(K.tensorflow_backend._get_available_gpus())
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

# Get the environment and extract the number of actions.
env = gym.make("FlappyBird-v0")
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

obs=env.reset()

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()

#obs.shape

model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
#print(model.summary())

class AtariProcessor(Processor):
	def process_observation(self, observation):
		assert observation.ndim == 3  # (height, width, channel)
		img = Image.fromarray(observation)
		#img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
		img=np.array(img.resize(INPUT_SHAPE))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
		img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]
		processed_observation = np.array(img)
		assert processed_observation.shape == INPUT_SHAPE
		return processed_observation.astype('uint8')  # saves storage in experience memory

	def process_state_batch(self, batch):
		# We could perform this processing step in `process_observation`. In this case, however,
		# we would need to store a `float32` array instead, which is 4x more memory intensive than
		# an `uint8` array. This matters if we store 1M observations.
		processed_batch = batch.astype('float32') / 255.
		return processed_batch

	def process_reward(self, reward):
		return np.clip(reward, -1., 1.)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

weights_filename = 'callbacks/dqn_flappy_weights.h5f'
checkpoint_weights_filename = 'callbacks/dqn_flappy_weights_{step}.h5f'
log_filename = 'dqn_flappy_log.json'

callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]

history=dqn.fit(env, callbacks=callbacks, nb_steps=1500000, log_interval=10000, verbose=2)
#44,856.951 seconds

plt.plot(history.history["episode_reward"])
plt.savefig('images/episode_reward_with_preprocessing.png', dpi=100)
plt.show()

plt.plot(history.history["nb_episode_steps"])
plt.savefig('images/nb_episode_steps_with_preprocessing.png', dpi=100)
plt.show()

env.reset()
dqn.test(env, nb_episodes=10, visualize=True)


model_json = model.to_json()
with open("models/model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("models/model.h5")

