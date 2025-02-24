import os
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split

root_path = f"{os.getcwd()}/rnn"

with open(f"{root_path}/data/pianoabc.txt", "r") as f:
    data = f.read()

def word_index(data):
    result = {}
    data = list(set(data))
    data.sort()
    for i, a in enumerate(data):
        result[a] = i 
    return result 

def text_to_num(data):
    result = []
    for _, txt in enumerate(data):
        result.append(word_index[txt])
    return result

word_index = word_index(data)
text_list = list(data)
text_to_num = text_to_num(text_list)

X = []
y = []

for i in range(len(text_list)):
    if i > len(text_list) - 26:
        X = np.array(X)
        y = np.array(y)
        break
    X.append(text_to_num[i:i+25])
    y.append(text_to_num[i+25])

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)    
trainX, testX = tf.one_hot(trainX, len(word_index)), tf.one_hot(testX, len(word_index))
trainY, testY = tf.one_hot(trainY, len(word_index)), tf.one_hot(testY, len(word_index))

inputs = tf.keras.Input(shape=(25, 31))
x = tf.keras.layers.LSTM(100)(inputs)
outputs = tf.keras.layers.Dense(len(word_index), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=10)