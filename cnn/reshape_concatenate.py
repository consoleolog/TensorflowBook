import os
import tensorflow as tf

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX.reshape((len(trainX), 28, 28, 1)).astype("float32") / 255.0
testX = testX.reshape((len(testX), 28, 28, 1)).astype("float32") / 255.0

trainY = tf.one_hot(trainY, 10)
testY = tf.one_hot(testY, 10)

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # (14, 14, 32)
x = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')(x)  # (28, 28, 32)
x = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='relu')(x)  # (28, 28, 1)

x1 = tf.keras.layers.Flatten()(inputs)  # (None, 28 * 28 = 784)
x1 = tf.keras.layers.Dense(28 * 28 * 1, activation="relu")(x1)
x1 = tf.keras.layers.Reshape((28, 28, 1))(x1)  # (28, 28, 1)

x = tf.keras.layers.Concatenate()([x, x1])  # (28, 28, 2)
x = tf.keras.layers.Flatten()(x)  # (None, 28 * 28 * 2 = 1568)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY))