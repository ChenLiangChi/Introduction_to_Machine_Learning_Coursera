import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # First convolutional layer with 32 filters, each of size 3x3
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # MaxPooling layer to downsample the feature maps by taking the maximum value in each 2x2 patch
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional layer with 64 filters, each of size 3x3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # MaxPooling layer to downsample the feature maps by taking the maximum value in each 2x2 patch
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional layer with 128 filters, each of size 3x3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    # Flatten the feature maps to prepare for fully connected layers
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Example usage of the CNN model
if __name__ == '__main__':
    # Example data: replace with your own dataset
    # Assuming images with 28x28 pixels and 1 channel (grayscale)
    input_shape = (28, 28, 1)
    num_classes = 10  # Replace with the number of classes in your dataset

    # Create the CNN model
    model = create_cnn_model(input_shape, num_classes)

    # Compile the model with an appropriate loss function, optimizer, and metrics
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the summary of the model architecture
    model.summary()

    # Train the model with your own data (not included in this example)
    # Replace 'train_data', 'train_labels', 'validation_data', and 'validation_labels' with your data
    # model.fit(train_data, train_labels, epochs=10, validation_data=(validation_data, validation_labels))
