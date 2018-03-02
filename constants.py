INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
INPUT_SHAPE_CNN = (WINDOW_LENGTH,) + INPUT_SHAPE
SEED = 123
ENV_NAME = "FlappyBird-v0"

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

