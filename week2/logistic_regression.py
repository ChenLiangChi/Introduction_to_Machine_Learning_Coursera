# Tensorflow 2 based
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Using numpy to create 100 random points used as a reference line 
x_train = np.random.rand(100).astype(np.float32)
y_train = 0.1 * x_train + 0.3

# Initializes the weight and bias to be trained
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Iteration to be run
num_epoch = 201

# Selets the optimizer(Stochastic Gradient descent (SGD) optimizer is used over here)
optimize = tf.keras.optimizers.SGD(learning_rate=0.2)

# Iterations of gradient descent, optimizes the weight and bias by using the MSE as loss function
for epoch in range(num_epoch):
    with tf.GradientTape() as g:
        y = W * x_train + b
        cross_entrophy = tf.reduce_mean(tf.square(y - y_train))
    gradients = g.gradient(cross_entrophy, [W, b])
    optimize.apply_gradients(zip(gradients, [W, b]))
    
    # Plot the results
    plt.ion()
    plt.show()
    if epoch % 5 == 0:
        tf.print("Iteration: ", epoch, "Estimated weights: ", W, "Estimated bias: ", b,
                 "Cross entrophy(loss): ", cross_entrophy)
        plt.plot(x_train, y_train, 'ro', label='Original line')
        plt.plot(x_train, W * x_train + b, label='Trained line')
        #plt.legend()
        plt.pause(0.05)
        if epoch == 200:
            plt.waitforbuttonpress(0)
            plt.close()
        
        

        
        
