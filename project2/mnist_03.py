import os

import numpy as np
import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

train_x.reshape( (60000, 28, 28 , 1) )
test_x.reshape( (10000, 28, 28 , 1) )

train_x = train_x / 255.0
test_x = test_x / 255.0

inputs = tf.keras.Input(shape=(28, 28, 1)) # color 사진이면 3
x = tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_03")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(f"{os.getcwd()}/project2/checkpoint/mnist_02")

eval_val = model.evaluate( np.array( test_x ), np.array( tf.one_hot(test_y, 10) ) )

print(eval_val)