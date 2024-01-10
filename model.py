from main import training_iterator, validation_iterator
from main import BATCH_SIZE
import tensorflow as tf
from tensorflow import keras


model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape = (128, 128, 3)))

model.add(tf.keras.layers.Conv2D(128, 3, strides = 2, padding = 'same', activation = 'relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

model.add(tf.keras.layers.Conv2D(256, 3, strides = 2, padding = 'same', activation = 'relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

model.compile(
    loss = tf.keras.losses.CategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001),
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)

history = model.fit(
    training_iterator,
    steps_per_epoch = training_iterator.samples / BATCH_SIZE,
    epochs = 20,
    validation_data = validation_iterator,
    validation_steps = validation_iterator.samples / BATCH_SIZE
)

print(model.summary())

