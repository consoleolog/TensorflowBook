import os 
import tensorflow as tf
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

root_path = f"{os.getcwd()}/chap01"

df = pd.read_csv(f"{root_path}/data/gpascore.csv")
df.dropna(inplace=True)

X = df[['admit', 'gre', 'gpa']].to_numpy()
y = df['rank'].to_numpy()

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

inputs = tf.keras.Input(shape=(None, 1, 3))
x = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dense(units=32, activation='relu')(x)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10)

p = model.predict( np.array([[ 380, 3, 3 ]]) )
print(p)