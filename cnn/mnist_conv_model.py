import os 
import tensorflow as tf 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
from sklearn.model_selection import train_test_split
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# trainX.shape (60000, 28, 28)
# trainY.shape (60000, )
trainX = trainX.reshape((len(trainX), 28,28,1))
testX = testX.reshape((len(testX), 28,28,1))

trainY = tf.one_hot(trainY, 10)
testY = tf.one_hot(testY, 10)


inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1))(inputs)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10)
model.evaluate(testX, testY)