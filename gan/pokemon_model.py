import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from PIL import Image

root_path = f"{os.getcwd()}/gan"

def image_num_list():
    result = []
    for filename in os.listdir(f"{root_path}/data/pokemon"):
        image = Image.open(f"{root_path}/data/pokemon/{filename}")
        result.append(np.array(image))
    return result

image_num_list = image_num_list()
image_num_list = np.divide(image_num_list, 255)


def discriminator(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="discriminator")
    return model 

def generator(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(4 * 4 * 256)(inputs)
    x = tf.keras.layers.Reshape((4, 4, 256))(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid")(x)  # RGB 채널
    model = tf.keras.Model(inputs, outputs, name="generator")
    return model



latent_dim = 100
generator = generator(latent_dim)
discriminator = discriminator((256, 256, 3))

discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

def GAN(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    gen_out = generator(inputs)
    disc_out = discriminator(gen_out)
    model = tf.keras.Model(inputs, disc_out, name="GAN")
    return model

GAN = GAN(latent_dim)
GAN.compile(optimizer="adam", loss="binary_crossentropy")

# 랜덤 노이즈 생성 및 이미지 출력
rndn_num = np.random.uniform(-1, 1, size=(8, latent_dim))
y_hat = generator.predict(rndn_num)

print(y_hat.shape)  # (8, height, width, channels)
breakpoint()