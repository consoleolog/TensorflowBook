import os 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

root_path = f"{os.getcwd()}/cnn/data"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f"{root_path}/train/",
    image_size=(64, 64),
    batch_size=32,
    seed=1234,
)
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f"{root_path}/valid/",
    image_size=(64, 64),
    batch_size=32,
    seed=1234
)

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize)
valid_dataset = valid_dataset.map(normalize)

def imshow(img):
    plt.imshow(img.astype("uint8"))
    plt.show()

inputs = tf.keras.Input(shape=(64, 64, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64, 64 ,3))(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(53, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, validation_data=(valid_dataset), epochs=10)