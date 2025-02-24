import os
import tensorflow as tf 
import numpy as np 

root_path = f"{os.getcwd()}/rnn"

with open(f"{root_path}/data/pianoabc.txt", "r") as f:
    data = f.read()

def word_index(data):
    result = {}
    data = list(set(data))
    data.sort()
    for i, a in enumerate(data):
        result[a] = i 
    return result 

def text_to_num(data):
    result = []
    for _, txt in enumerate(data):
        result.append(word_index[txt])
    return result

word_index = word_index(data)
text_list = list(data)
text_to_num = text_to_num(text_list)

inputs = text_to_num[117:117+25]
inputs = np.array(inputs)
inputs = tf.one_hot(inputs, len(word_index))

model = tf.keras.models.load_model(f"{root_path}/models/music")
inputs = np.expand_dims(inputs, axis=0)

result = []
breakpoint()
for i in range(1000):
    predicts = model.predict(inputs)
    result.append(np.argmax(predicts))
    
        
    