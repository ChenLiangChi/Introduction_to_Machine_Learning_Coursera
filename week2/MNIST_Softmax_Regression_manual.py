# Tensorflow 2 based, without using sequential model
import numpy as np
import tensorflow as tf
from tqdm import trange  

mnist = tf.keras.datasets.mnist

# Define the diemensions of MNIST dataset: 28*28=784 pixels with 10 categories (digits)
mnist_dim = 28 * 28
mnist_class = 10

# Load the MNIST dataset
# shape of x_train: (60000, 28, 28); y_train: (60000)
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalize the input data to values between 0 and 1 
# (As the reason that original MNIST values are between 0 and 255)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data to a flat vector (784-dimensional) for each image
x_train = x_train.reshape(-1, mnist_dim)
x_test = x_test.reshape(-1, mnist_dim)

# Convert the target labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, mnist_class)
y_test = tf.keras.utils.to_categorical(y_test, mnist_class)

# Define the weight and bias
W = tf.Variable(tf.zeros([mnist_dim, mnist_class]), dtype=tf.float32)
b = tf.Variable(tf.zeros([mnist_class]), dtype=tf.float32) 

@tf.function
# Model based on Softmax function
def Softmax_model(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)
# Loss function: Categorical cross-entropy
def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
# Define the accuracy metric
def accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

# Define the optimizer (SGD)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

# Define training loops
epochs = 1000
batch_size = 100
num_batches = len(x_train) #60000

for epoch in trange(epochs):
    start_idx = np.random.randint(0, num_batches)
    end_idx = start_idx + batch_size
    x_batch = x_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]
    
    with tf.GradientTape() as g:
        predictions = Softmax_model(x_batch)
        loss = cross_entropy(y_batch, predictions)
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Evaluate the model    
train_accuracy = accuracy(y_train, Softmax_model(x_train))
test_accuracy = accuracy(y_test, Softmax_model(x_test))
    
print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")


