import os 
import tensorflow as tf 
import numpy as np 



normalization_layer = tf.keras.layers.Normalization(mean=2.0, variance=1.0)

def StringLookup(vocab_list):
    return tf.keras.layers.StringLookup(
        vocabulary=vocab_list,
        num_oov_indices=0,
        output_mode="one_hot"
    )
    
def OneHotLayer(tokens):
    return tf.keras.layers.CategoryEncoding(
        num_token=tokens,
        output_mode="one_hot"
    )
    
string_lookup_layer = StringLookup([])
embedding = tf.keras.layers.Embedding()
embedding(string_lookup_layer([]))