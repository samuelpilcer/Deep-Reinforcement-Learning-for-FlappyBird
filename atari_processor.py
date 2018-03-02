import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rl.core import Processor

from constants import *


class AtariProcessor(Processor):
    def __init__(self, input_shape, preprocess_images):
        self.INPUT_SHAPE = input_shape
        self.preprocess_images = preprocess_images

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        if self.preprocess_images:
            img = img.crop(CROPPED_SIZE)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]
            img = Image.fromarray(img, 'L')
            img = img.resize(SHRUNKEN_SHAPE)
        else:
            img = img.resize(self.INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.T.shape == self.INPUT_SHAPE
        return processed_observation.T.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
