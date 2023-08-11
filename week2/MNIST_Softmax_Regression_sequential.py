# Tensorflow 2 based
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist

# Load the MNIST dataset
# shape of x_train: (60000, 28, 28); y_train: (60000)
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalize the input data to values between 0 and 1 
# (As the reason that original MNIST values are between 0 and 255)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data to a flat vector (784-dimensional) for each image
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert the target labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the softmax regression model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='softmax', input_shape=(28 * 28,))
])

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_accuracy)