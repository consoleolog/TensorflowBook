import os
import shutil
import time

from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

def create_data():
    if not os.path.exists(f"{os.getcwd()}/project2/data/dog"):
        os.mkdir(f"{os.getcwd()}/project2/data/dog")
    if not os.path.exists(f"{os.getcwd()}/project2/data/cat"):
        os.mkdir(f"{os.getcwd()}/project2/data/cat")

    for i, filename in enumerate(os.listdir(f"{os.getcwd()}/project2/data/train")):
        if 'cat' in filename:
            shutil.copy(f"{os.getcwd()}/project2/data/train/{filename}", f"{os.getcwd()}/project2/data/cat/{filename}")

        if 'dot' in filename:
            shutil.copy(f"{os.getcwd()}/project2/data/train/{filename}", f"{os.getcwd()}/project2/data/dog/{filename}")

img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # 회전
    zoom_range=0.15,        # 확대
    width_shift_range=0.2,  # 이동
    height_shift_range=0.2, # 이동
    shear_range=0.15,       # 굴절
    horizontal_flip=True,   # 가로 반전
    fill_mode="nearest"
)

train_ds = img_generator.flow_from_directory(
    directory=f"{os.getcwd()}/project2/data/cat_and_dog",
    class_mode='binary', # categorical
    shuffle=True,
    seed=123,
    color_mode='rgb',
    batch_size=32,
    target_size=(64, 64)
)


def get_data():
    x_train = []
    y_train = []

    for i in train_ds:
        x_train.append(i[0])
        y_train.append(i[1])

    return {
        "trainX": np.array(x_train),
        "trainY": np.array(y_train)
    }

result = get_data()

x, y = result["trainX"], result["trainY"]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

inputs = tf.keras.Input(shape=(64,64,3))
x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(inputs)
x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
x = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same' ,activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same' ,activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='kaggle_01')

LOG_PATH = f"{os.getcwd()}/project2/logs/{model.name}_{time.strftime('%Y_%m%d_%H_%M_%S')}"

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)

es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min' # max
    )

if not os.path.exists(f"{os.getcwd()}/project2/checkpoint/{model.name}"):
    os.mkdir(f"{os.getcwd()}/project2/checkpoint/{model.name}")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{os.getcwd()}/project2/checkpoint/{model.name}/{model.name}", # 변수에 epoch 가능
    monitor='val_acc',
    mode='max',
    save_weights_only=True,
    save_freq='epoch',
)

train_x = train_x / 255.0
test_x = test_x / 255.0

model.summary()
model.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

model.fit(np.array(train_x), np.array(train_y),
          validation_data=(np.array(test_x), np.array(test_y)),
          epochs=5,
          batch_size=32,
          callbacks=[tensorboard, es, checkpoint])

model.save(f"{os.getcwd()}/project2/models/{model.name}")
# sparse_categorical_accuracy
