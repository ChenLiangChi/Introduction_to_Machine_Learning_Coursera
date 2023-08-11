# Tensorflow 2 based
import tensorflow as tf
from tqdm import trange

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the diemensions of MNIST dataset: 28*28=784 pixels with 10 categories (digits)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data to a flat vector (784-dimensional) for each image
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert the target labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def multi_layer_perceptron(input_dim, hidden1_unit, hidden2_unit, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_dim,)),
        tf.keras.layers.Dense(hidden1_unit, activation='relu'),
        tf.keras.layers.Dense(hidden2_unit, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model

input_dim = 784     # 28*28
output_dim = 10     # 10 classes
hidden1_unit = 500  
hidden2_unit = 100
learning_rate = 0.1

model = multi_layer_perceptron(input_dim, hidden1_unit, hidden2_unit, output_dim)

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=0)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy: {0}'.format(test_accuracy))
