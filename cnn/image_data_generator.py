import os 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

root_path = f"{os.getcwd()}/cnn/data"

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # 회전
    zoom_range=0.15,        # 확대
    width_shift_range=0.2,  # 이동
    height_shift_range=0.2, # 이동
    shear_range=0.15,       # 굴절
    horizontal_flip=True,   # 가로 반전
    fill_mode="nearest"
)

train_dataset = train_generator.flow_from_directory(
    directory=f"{root_path}/train/",
    class_mode="categorical",
    shuffle=True,
    seed=1234,
    color_mode="rgb",
    batch_size=32,
    target_size=(64, 64)
)
valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_dataset = valid_generator.flow_from_directory(
    directory=f"{root_path}/valid/",
    class_mode="categorical",
    shuffle=True,
    seed=1234,
    color_mode="rgb",
    batch_size=32,
    target_size=(64, 64)
)

def imshow(img):
    plt.imshow(img.astype("uint8"))
    plt.show()

inputs = tf.keras.Input(shape=(64, 64, 3))
x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64, 64 ,3))(inputs)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(53, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, validation_data=(valid_dataset), epochs=10)