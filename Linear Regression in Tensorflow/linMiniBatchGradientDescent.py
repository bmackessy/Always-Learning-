import tensorflow as tf
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.preprocessing import StandardScaler


n_epochs = 1000
learning_rate = 0.01

dataset = load_breast_cancer()

# m is length of dataset, n is # of atts
m, n = dataset.data.shape

# Scale Data with StandardScalar()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.data)

# m is length of dataset, n is # of atts
m, n = dataset.data.shape

batch_size = 100
num_batches = int(np.ceil(m/batch_size))


# Just appends a a column of '1.'s to beginnging of table
scaled_data_plus_bias = np.c_[np.ones((m, 1)), scaled_data]

X = tf.placeholder(tf.float32, shape=(batch_size, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1),name="theta")

# Formulas for preducing predictions and error metrics
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")


# Operation that adjusts theta values based on gradients & learning rate
# training_op = theta.assign(theta - learning_rate * gradients)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * num_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_data_plus_bias[indices]
    y_batch = dataset.target.reshape(-1, 1)[indices]

    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)
    best_theta = None
    best_mse = None
    for epoch in range(n_epochs):
        
        for batch_index in range(num_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            _, best_mse = sess.run([training_op,mse], feed_dict={X: X_batch, y: y_batch})
        
        if epoch % 100 == 0:
            print("Epoch: " + str(epoch))
            print("MSE " + str(best_mse))
       
    print(theta.eval())
        

