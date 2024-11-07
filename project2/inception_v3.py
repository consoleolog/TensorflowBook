import os
import time
import numpy as np

from keras.applications import InceptionV3
import tensorflow as tf
from sklearn.model_selection import train_test_split

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f"{os.getcwd()}/project2/data/cat_and_dog",
    image_size=(64,64),
    batch_size=32,
)

def get_data():
    x_train = []
    y_train = []

    for images, labels in train_ds.unbatch():
        x_train.append(images.numpy())
        y_train.append(labels.numpy())

    train_x, test_x, train_y, test_y = train_test_split(np.array(x_train), np.array(y_train), test_size=0.2, random_state=42)

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    return {
        "trainX": train_x,
        "trainY": train_y,
        "testX": test_x,
        "testY": test_y,
    }

trainX, trainY, testX, testY = get_data()["trainX"], get_data()["trainY"], get_data()["testX"], get_data()["testY"]

inception_v3 = InceptionV3( input_shape=(150, 150, 3), include_top=False, weights=None )
inception_v3.load_weights(f'{os.getcwd()}/project2/models/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

for i, layer in enumerate(inception_v3.layers):
    # w 값 update X
    layer.trainable = False

mixed7 = inception_v3.get_layer("mixed7")

inputs = tf.keras.layers.Flatten()(mixed7.output)
x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
# x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
# x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_inception")

# tf.keras.utils.plot_model(inception_v3,to_file=f"{os.getcwd()}/project2/img/{model.name}.png", show_shapes=True, show_layer_names=True)

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array( trainX ), np.array( trainY ),
          validation_data=(np.array( testX ), np.array(testY)),
          epochs=10, callbacks=[es,tensorboard,checkpoint])

# model.evaluate( np.array( testX ), np.array( tf.one_hot(testY, 10) ) )

model.save(f"{os.getcwd()}/project2/models/{model.name}")