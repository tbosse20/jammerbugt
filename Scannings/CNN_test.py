import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate some example data (replace this with your own dataset)
num_samples = 1000
image_size = 25
num_classes = 10

x_train = np.random.random((num_samples, image_size, image_size, 3))
y_train = np.random.randint(num_classes, size=(num_samples,))

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Save the model weights
model.save_weights('cnn_model_weights.h5')
print("Model weights saved.")
