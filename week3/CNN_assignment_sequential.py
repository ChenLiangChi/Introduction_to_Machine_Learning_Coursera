import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Define the dimensions of MNIST dataset: 28*28=784 pixels with 10 categories (digits)
mnist_dim = 28 * 28
mnist_class = 10

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data to values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the target labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, mnist_class)
y_test = tf.keras.utils.to_categorical(y_test, mnist_class)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 128

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Save the model under the given path
model.save('/Users/liangchichen/Desktop/intro_to_ML/week3/model_checkpoint.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: ", test_accuracy)
