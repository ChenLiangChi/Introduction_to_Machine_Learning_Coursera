import tensorflow as tf
import numpy as np
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

# Reshape the input data to 4D representing gray-scale image
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

# Convert the target labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, mnist_class)
y_test = tf.keras.utils.to_categorical(y_test, mnist_class)

def cnn_layer1(x_cnn):
    conv1_preact = tf.nn.conv2d(x_cnn, W_cnn1, strides=[1, 1, 1, 1], padding='SAME') + b_cnn1
    return tf.nn.relu(conv1_preact)
def cnn_layer2(cnn2_in):
    conv2_preact = tf.nn.conv2d(cnn2_in, W_cnn2, strides=[1, 1, 1, 1], padding='SAME') + b_cnn2
    return tf.nn.relu(conv2_preact)
def max_pool(pool_in):
    return tf.nn.max_pool(pool_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def full_layer1(full1_in):
    return tf.nn.relu(tf.matmul(full1_in, W_full1) + b_full1)
def full_layer2(full2_in):
    return tf.nn.softmax(tf.matmul(full2_in, W_full2) + b_full2)
def accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))
def model_test(train_input):
    max_conv2_test = max_pool(cnn_layer2(max_pool(cnn_layer1(train_input))))
    max_conv2_test = tf.reshape(max_conv2_test, [10000, 3136])
    return full_layer2(full_layer1(max_conv2_test))

W_cnn1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=1, name='weight_cnn1'))
b_cnn1 = tf.Variable(tf.zeros([32], name='bias_cnn1'))

W_cnn2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=1, name='weight_cnn2'))
b_cnn2 = tf.Variable(tf.zeros([64], name='bias_cnn2'))

W_full1 = tf.Variable(tf.random.truncated_normal([3136, 256], stddev=1, name='weight_full1'))
b_full1 = tf.Variable(tf.zeros([256], name='bias_full1'))

W_full2 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=1, name='weight_full2'))
b_full2 = tf.Variable(tf.zeros([10], name='bias_full2'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

epochs = 10000
batch_size = 64

# Create chaeckpoints for weights and biases
checkpoint = tf.train.Checkpoint(W_cnn1=W_cnn1, b_cnn1=b_cnn1,
                                 W_cnn2=W_cnn2, b_cnn2=b_cnn2,
                                 W_full1=W_full1, b_full1=b_full1,
                                 W_full2=W_full2, b_full2=b_full2)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as g:
        conv1 = cnn_layer1(x_batch)
        max_conv1 = max_pool(conv1)
        conv2 = cnn_layer2(max_conv1)
        max_conv2 = max_pool(conv2)
        max_conv2 = tf.reshape(max_conv2, [batch_size, 3136])
        full1 = full_layer1(max_conv2)
        full2 = full_layer2(full1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=full2, labels=y_batch))

    gradients = g.gradient(loss, [W_cnn1, b_cnn1, W_cnn2, b_cnn2, W_full1, b_full1, W_full2, b_full2],
                           unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(gradients, [W_cnn1, b_cnn1, W_cnn2, b_cnn2, W_full1, b_full1, W_full2, b_full2]))
    return full2

for epoch in trange(epochs):    
    step = np.random.randint(0, len(x_train)-batch_size)
    
    x_batch = x_train[step:step + batch_size]
    y_batch = y_train[step:step + batch_size]

    full2 = train_step(x_batch, y_batch)

# Save the checkpoint under the given path
checkpoint.save('/Users/liangchichen/Desktop/intro_to_ML/week3/')

#tf.print("Train accuaracy: ", accuracy(y_train, full2))
tf.print("Test accuaracy: ", accuracy(y_test, model_test(x_test)))

