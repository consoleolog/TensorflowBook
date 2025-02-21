import os
import time

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(train_x[0])
# plt.show()

# print(train_x.shape)

inputs = tf.keras.Input(shape=(28, 28, 1)) # color 사진이면 3
x = tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_02")

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


tf.keras.utils.plot_model(model,to_file=f"{os.getcwd()}/project2/img/{model.name}.png", show_shapes=True, show_layer_names=True)

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

train_x.reshape( (60000, 28, 28 , 1) )
test_x.reshape( (10000, 28, 28 , 1) )

train_x = train_x / 255.0
test_x = test_x / 255.0



model.fit(np.array( train_x ), np.array( tf.one_hot(train_y, 10) ),
          validation_data=(np.array( test_x ), np.array(tf.one_hot(test_y, 10) )),
          epochs=10, callbacks=[es,tensorboard,checkpoint])

model.evaluate( np.array( test_x ), np.array( tf.one_hot(test_y, 10) ) )

model.save(f"{os.getcwd()}/project2/models/{model.name}")
