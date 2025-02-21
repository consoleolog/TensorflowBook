import os

import tensorflow as tf
import keras_tuner as kt

(img_train, label_train), (img_test, label_test) = tf.keras.datasets.mnist.load_data()

img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    model = tf.keras.Sequential()

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(optimizer= tf.keras.optimizers.Adam(lr=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory=f"{os.getcwd()}/project0/model",
    project_name='hyperparameter_tune',
)


tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
model.fit(img_train, label_train, epochs = 10, validation_data = (img_test, label_test))