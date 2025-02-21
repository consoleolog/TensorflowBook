import os, sys
import tensorflow as tf 
import matplotlib.pyplot as plt 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils 

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# trainX.shape (60000, 28, 28)
# trainY.shape (60000, )
trainY = tf.one_hot(trainY, 10)
testY = tf.one_hot(testY, 10)


inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

root_path = f"{os.getcwd()}/cnn"
checkpoint = utils.checkpoint(f"{root_path}/checkpoint")

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=10, callbacks=[checkpoint])