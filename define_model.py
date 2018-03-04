from keras.layers import Dense, Flatten, Conv2D, Permute, Conv2DTranspose, Input, concatenate, Reshape, Cropping1D
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from atari_processor import AtariProcessor
from constants import *


def load_model(filename='models/model_raw_images.'):
    model_json = open(filename + 'json', 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + 'h5')
    print("Loaded model from disk")
    return loaded_model


def get_cnn_model(input_shape=INPUT_SHAPE_CNN, nb_actions=2):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def get_agent_from_model(model, nb_actions, input_shape):
    memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor(input_shape=input_shape, preprocess_images=PREPROCESS)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=EPS_MAX, value_min=EPS_MIN,
                                  value_test=EPS_TEST, nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor,
                   batch_size=BATCH_SIZE, nb_steps_warmup=50000, gamma=GAMMA, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])
    return dqn


def adapt_model_to_alife(model, input_shape=(10, 1), vision_shape=(1, 3, 3), energy_shape=(1, 1, 1)):
    bug_input = Input(shape=input_shape)

    model_vision = Sequential()
    model_vision.add(Cropping1D(cropping=(0, 1), input_shape=input_shape))
    model_vision.add(Reshape(target_shape=vision_shape))
    model_vision.add(Conv2DTranspose(4, (SHRUNKEN_SHAPE[0], int(SHRUNKEN_SHAPE[1] / 2) - 1),
                                     input_shape=vision_shape, padding='valid'))
    model_vision.add(Permute((3, 1, 2)))
    vision_out = model_vision(bug_input)

    model_energy = Sequential()
    model_energy.add(Cropping1D(cropping=(9, 0), input_shape=input_shape))
    model_energy.add(Reshape(target_shape=energy_shape))
    model_energy.add(Conv2DTranspose(4, (SHRUNKEN_SHAPE[0], int(SHRUNKEN_SHAPE[1] / 2) - 1),
                                     input_shape=energy_shape, padding='valid'))
    model_energy.add(Permute((3, 1, 2)))
    energy_out = model_energy(bug_input)

    merged_tensor = concatenate([vision_out, energy_out], axis=-1)

    model.pop()
    model.add(Reshape((1, 512, 1)))
    model.add(Conv2D(SPEED_ACTION_POSSIBILITIES, (1, 512 - ANGLE_ACTION_POSSIBILITIES + 1), name='conv_2d_alife'))

    output = model(merged_tensor)

    complete_model = Model(inputs=bug_input, outputs=output)

    return complete_model
