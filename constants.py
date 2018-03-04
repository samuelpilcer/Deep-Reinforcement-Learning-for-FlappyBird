INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
INPUT_SHAPE_CNN = (WINDOW_LENGTH,) + INPUT_SHAPE
SEED = 123
ENV_NAME = "FlappyBird-v0"

GAMMA = 0.99
EPS_MAX = 1.
EPS_MIN = 0.1
EPS_TEST = 0.05
EPS_DECAY = 0.995
LEARNING_RATE = .00025
BATCH_SIZE = 32
MEMORY = 10000
FIT_INTERVAL = 100

PREPROCESS = True
CROPPED_SIZE = (0, 0, 288, 370)
IMAGE_SIZE_DIVIDER = 8
SHRUNKEN_SHAPE = tuple([int(i / IMAGE_SIZE_DIVIDER) for i in CROPPED_SIZE[2:]])

CHECKPOINT_WEIGHTS_FILENAME = 'callbacks/dqn_flappy_weights_{step}_raw_images.h5f'
LOG_FILENAME = 'dqn_flappy_log_raw_images.json'
MODEL_JSON_FILENAME = "models/model_raw_images.json"
MODEL_WEIGHTS_FILENAME = "models/model_raw_images.h5"

if PREPROCESS:
    CHECKPOINT_WEIGHTS_FILENAME = 'callbacks/dqn_flappy_weights_{step}_preprocessed_images.h5f'
    LOG_FILENAME = 'dqn_flappy_log_preprocessed_images.json'
    MODEL_JSON_FILENAME = "models/model_preprocessed_images.json"
    MODEL_WEIGHTS_FILENAME = "models/model_preprocessed_images.h5"


SPEED_STEP = 1
ANGLE_STEP = 15

SPEED_ACTION_POSSIBILITIES = int(21 / SPEED_STEP)
ANGLE_ACTION_POSSIBILITIES = int(90 / ANGLE_STEP) + 1
