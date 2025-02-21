import tensorflow as tf

# sample data
height = 170
shoes = 260

# sample a , b
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# y ( 신발 사이즈 ) = height * a + b
# loss 함수
def loss_func():
    # (예측값 - 실제값)** 2 -> tf.square(예측값 - 실제값)
    return tf.square(260 - ( height * a + b) )

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(1000):
    # 업데이트 할 변수 목록 [a, b]
    opt.minimize(loss_func, var_list=[a, b])

print(a)
print(b)
def exp(size):
    return a * size + b

print(exp(150))


