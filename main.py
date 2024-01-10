import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

DIRECTORY_TRAIN = 'archive-5/train'
DIRECTORY_VALID = 'archive-5/test'
CLASS_MODE = 'categorical'
COLOR_MODE = 'rgb'
TARGET_SIZE = (128, 128)
BATCH_SIZE = 64

training_data_generator = ImageDataGenerator(
    rescale = 1.0 / 255,
    rotation_range = 20,
    zoom_range = 0.2,
    height_shift_range = 0.2,
    width_shift_range = 0.2,
    horizontal_flip =  True  
)

training_iterator = training_data_generator.flow_from_directory(
    directory = DIRECTORY_TRAIN,
    class_mode = CLASS_MODE,
    color_mode = COLOR_MODE,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE
)

validation_data_generator = ImageDataGenerator(
    rescale = 1.0/255
)


validation_iterator = validation_data_generator.flow_from_directory(
    directory = DIRECTORY_VALID,
    class_mode = CLASS_MODE,
    color_mode = COLOR_MODE,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE
)