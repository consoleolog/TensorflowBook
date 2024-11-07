import os
import tensorflow as tf
import time

def get_tf_callbacks(model, path):
    tf.keras.utils.plot_model(model, to_file=f"{os.getcwd()}/project2/img/{model.name}.png", show_shapes=True,
                              show_layer_names=True)

    LOG_PATH = f"{os.getcwd()}/project2/logs/{model.name}_{time.strftime('%Y_%m%d_%H_%M_%S')}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'  # max
    )

    if not os.path.exists(f"{os.getcwd()}/project2/checkpoint/{model.name}"):
        os.mkdir(f"{os.getcwd()}/project2/checkpoint/{model.name}")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{os.getcwd()}/project2/checkpoint/{model.name}/{model.name}",  # 변수에 epoch 가능
        monitor='val_acc',
        mode='max',
        save_weights_only=True,
        save_freq='epoch',
    )

# tensorboard --logdir logs