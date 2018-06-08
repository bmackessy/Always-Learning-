'''
Author: Brian Mackessy
Created: 6/6/18
Last Edited: 6/7/18
Description: Class for logistic regression. Formatted similarly to the logisitc regresson in Scikit-Learn.
             See tensorboard for training visualizations
'''


import tensorflow as tf
from sklearn.datasets import make_moons
import numpy as np
from datetime import datetime


class Logit:
    theta = []
    

    def fetch_batch(self, epoch, batch_index, batch_size, num_batches, X_data, y_data, m):
        np.random.seed(epoch * num_batches + batch_index)  
        indices = np.random.randint(m, size=batch_size)
        X_batch = X_data[indices]
        y_batch = y_data.reshape(-1, 1)[indices]

        return X_batch, y_batch


    
    def fit(self, X_data, y_data, n_epochs=200, batch_size=50, learning_rate=0.1):
        # m is length of dataset, n is # of atts
        m, n = X_data.data.shape

        num_batches = int(np.ceil(m/batch_size))

        # Just appends a a column of '1.'s to beginnging of table
        X_data_plus_bias = np.c_[np.ones((m, 1)), X_data]

        X = tf.placeholder(tf.float32, shape=(batch_size, n+1), name="X")
        y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")
        theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1),name="theta")

        prediction = tf.sigmoid(tf.matmul(X, theta))
        loss = tf.losses.log_loss(y, prediction)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)


        # Setup variables for visualization
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        logdir = "{}/run-{}/".format(root_logdir, now)
        loss_summary = tf.summary.scalar('Loss', loss)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            curr_loss = 0
            for epoch in range(n_epochs):
                for batch_index in range(num_batches):
                    X_batch, y_batch = self.fetch_batch(epoch, batch_index, batch_size, num_batches, X_data_plus_bias, y_data, m)
                    if batch_index % 10 == 0:
                        summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                        step = epoch * num_batches + batch_index
                        file_writer.add_summary(summary_str, step)

                    _, curr_loss = sess.run([training_op, loss], feed_dict = {X: X_batch, y: y_batch})
       
            
            self.theta = theta.eval()
            file_writer.close()


    def decisionBoundary(self, val):
        if val >= 0.5:
            return 1
        else:
            return 0
   
    # Will predict any number of instances in [1,inf]
    def predict(self, X_instance):

        m, n = (None, None)
        x_instance_plus_bias = None

        if len(X_instance.shape) == 1:
            m = 1
            n = X_instance.shape[0]
            X_instance_plus_bias = np.insert(X_instance, 0, 1.0)     
        else:
            m, n = X_instance.shape
            X_instance_plus_bias = np.c_[np.ones((m, 1)), X_instance]


        X_instance_plus_bias = np.reshape(X_instance_plus_bias, (m, n+1))

        X = tf.Variable(X_instance_plus_bias, dtype=tf.float32, name="X")
        theta = tf.Variable(self.theta, dtype=tf.float32, name="theta")
        
        t = tf.matmul(X, theta)
        prediction = tf.sigmoid(t)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            return [self.decisionBoundary(i) for i in prediction.eval()] 
        
            

# The class contains a general Logistic Regression model
# Any dataset can be plugged in. The moon dataset is just a common example
def main():
    m = 1000
    X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

    # Example of how to use the class
    logit = Logit()
    logit.fit(X_moons,y_moons)
    
    # Predicting all 1,000 moon instances
    predictons = logit.predict(X_moons)

if __name__== "__main__":
    main()





