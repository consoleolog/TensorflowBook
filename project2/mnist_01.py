import os
import time

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(train_x[0])
# plt.show()

# print(train_x.shape)

inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_01")

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

LOG_PATH = f"{os.getcwd()}/project2/logs/{model.name}_{time.strftime('%Y_%m%d-%H_%M_%S')}"

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)

es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min' # max
    )

train_x = train_x / 255.0
test_x = test_x / 255.0

model.fit(np.array( train_x ), np.array( tf.one_hot(train_y, 10) ), epochs=5, callbacks=[es,tensorboard])
model.save(f"{os.getcwd()}/project2/{model.name}")
