import os 
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

root_path = f"{os.getcwd()}/rnn"
# urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', f'{root_path}/data/shopping.txt')

data = pd.read_table(f"{root_path}/data/shopping.txt", names=["rating", "review"])

# Setting Label
data["label"] = np.where(data["rating"] > 3, 1, 0)
data["len"] = data["review"].str.len()

# 특수, 중복 문자 제거
data["review"] = data["review"].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]", "")
data.drop_duplicates(subset=["review"], inplace=True)

text_list = data["review"].tolist()
tokenizer = Tokenizer(char_level=True, oov_token="<OOV>")
tokenizer.fit_on_texts(text_list)

# 문자 리스트를 정수로 변환 
train_sequences = tokenizer.texts_to_sequences(text_list)

X = pad_sequences(train_sequences, maxlen=data["len"].max())
y = np.array(data["label"].tolist())

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
inputs = tf.keras.Input(shape=(140, None))
x = tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16)(inputs)
x = tf.keras.layers.LSTM(128)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1)
