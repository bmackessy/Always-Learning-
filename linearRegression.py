import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np


housing = fetch_california_housing()


# m is length of dataset, n is # of atts
m, n = housing.data.shape

# Just appends a a column of '1.'s to beginnging of table
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.Variable(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.Variable(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

# tf.transpose
# tf.matmul
# tf.matrix_inverse
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(theta.eval())
























    
     
    
