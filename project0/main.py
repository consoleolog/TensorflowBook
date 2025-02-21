import tensorflow as tf

X = tf.constant( [ [10,20,30,40] ])
W = tf.constant( [ [100,200,300,400] ])

print(tf.add(X,W))

node1 = tf.matmul( X, tf.transpose(W) )
print(node1)

zeros = tf.zeros([2, 2, 3])
print(zeros)

tf_variable = tf.Variable(1.0)
print(tf_variable)
print(tf_variable.numpy())
tf_variable.assign(2)
print(tf_variable)




