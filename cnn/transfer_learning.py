import os, sys
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from keras.applications.inception_v3 import InceptionV3
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils 

root_path = f"{os.getcwd()}/cnn/"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f"{root_path}/data/train/",
    image_size=(150, 150),
    batch_size=32,
    seed=1234,
)
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f"{root_path}/data/valid/",
    image_size=(150, 150),
    batch_size=32,
    seed=1234
)

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize)
valid_dataset = valid_dataset.map(normalize)

inception_v3 = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
inception_v3.load_weights(f"{root_path}/models/inception_v3.h5")

# Finetune
unfreeze = False 
for _, layer in enumerate(inception_v3.layers):
    layer.trainable = False 
    if layer.name == "mixed6":
        unfreeze = True  
    if unfreeze:
        layer.trainable = True 

mixed7 = inception_v3.get_layer("mixed7") # output (7, 7, 768)
inputs = tf.keras.layers.Flatten()(mixed7.output)
x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(53, activation="softmax")(x)
model = tf.keras.Model(inception_v3.input, outputs)
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, validation_data=(valid_dataset), epochs=10)