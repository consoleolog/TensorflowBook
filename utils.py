import os 
import time 
import tensorflow as tf 

def earlystopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

def tensorboard(log_path=f"{os.getcwd()}/logs"):
    curr_time = time.strftime("%Yy%mm%dd%Hh%Mm%Ss")
    save_path = f"{log_path}/{curr_time}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return tf.keras.callbacks.TensorBoard(log_dir=save_path)

def save_model_as_png(model, save_path=f"{os.getcwd()}/models/images"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tf.keras.utils.plot_model(model, f"{save_path}/{model.name}.png")
    