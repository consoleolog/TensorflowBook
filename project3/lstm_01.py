import os
import time

import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split

text = open(f"{os.getcwd()}/project3/data/pianoabc.txt", "r").read()

word_bag = list(set(text))
word_bag.sort()

def convert_text_to_num(txt_list):
    text_to_num = {}

    for i, txt in enumerate(txt_list):
        text_to_num[txt] = i

    return text_to_num

def convert_num_to_text(txt_list):

    num_to_text = {}

    for i, txt in enumerate(txt_list):
        num_to_text[i] = txt

    return num_to_text

def create_num_list():
    num_list = []

    for i, txt in enumerate(text):
        num_list.append( convert_text_to_num(word_bag)[txt] )

    return num_list

def get_data():

    data = create_num_list()

    X = []
    Y = []
    for i, num in enumerate(data):
        if i > len(data) - 26:
            break

        X.append(data[i:i+25])
        Y.append(data[i+25])
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

    return {
        "trainX": np.array(train_x),
        "trainY": np.array(train_y),
        "testX": np.array(test_x),
        "testY": np.array(test_y),
    }

trainX, trainY, testX, testY = get_data()["trainX"], get_data()["trainY"], get_data()["testX"], get_data()["testY"]

trainY, testY = tf.one_hot(trainY, depth=len(word_bag)), tf.one_hot(testY, depth=len(word_bag))


inputs = tf.keras.Input(shape=(25, 31))
x = tf.keras.layers.LSTM(100)(inputs)
outputs = tf.keras.layers.Dense(len(word_bag), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_abc")

tf.keras.utils.plot_model(model,to_file=f"{os.getcwd()}/project3/img/{model.name}.png", show_shapes=True, show_layer_names=True)

LOG_PATH = f"{os.getcwd()}/project3/logs/{model.name}_{time.strftime('%Y_%m%d_%H_%M_%S')}"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)

es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min' # max
    )

if not os.path.exists(f"{os.getcwd()}/project3/checkpoint/{model.name}"):
    os.mkdir(f"{os.getcwd()}/project3/checkpoint/{model.name}")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{os.getcwd()}/project3/checkpoint/{model.name}/{model.name}", # 변수에 epoch 가능
    monitor='val_acc',
    mode='max',
    save_weights_only=True,
    save_freq='epoch',
)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=10, callbacks=[es,tensorboard,checkpoint])

model.save(f"{os.getcwd()}/project3/models/{model.name}")
