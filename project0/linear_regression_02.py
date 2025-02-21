import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

# sample a, b
a = tf.Variable(0.0)
b = tf.Variable(0.0)

def loss_func(a, b):
    # ( 예측1 - 실제1 )^2 + ( 예측2 - 실제2 )^2
    # tf.keras.losses.mse(실제값, 예측값)
    # 예측값
    predict_val = train_x * a + b
    return tf.keras.losses.mse(train_y, predict_val)

opt = tf.optimizers.Adam(learning_rate=0.1)

for i in range(10000):
    opt.minimize(lambda: loss_func(a, b), var_list=[a, b])

print(a)
print(b)