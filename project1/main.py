import os

import numpy as np
import pandas as pd
import tensorflow as tf

def get_train_data():
    df = pd.read_csv(f"{os.getcwd()}/project_1/data/gpascore.csv")
    df.dropna(inplace=True)

    train_x = []
    train_y = []

    for i, data in df.iterrows():
        train_x.append([ data["gre"], data["gpa"], data["rank"] ])
        train_y.append([ data["admit"] ])

    return {
        "trainX": train_x,
        "trainY": train_y
    }

inputs = tf.keras.Input(shape=(None,3))
x = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dense(units=32, activation='relu')(x)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

result = get_train_data()

model.fit(x=np.array( result["trainX"] ), y=np.array( result["trainY"] ), epochs=100)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=256, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=8, activation='relu'),
#     tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# result = get_train_data()
#
# model.fit(x=np.array( result["trainX"] ), y=np.array( result["trainY"] ), epochs=100)