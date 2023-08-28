import random

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

label_classes = ["brown", "dark blue", "kaki", "light blue", "yellow"]


def load_data(dataset_directory, image_size):
    print('Loading data..')

    # List of class names (subdirectory names)
    class_names = os.listdir(dataset_directory)

    # Initialize lists to store images and corresponding labels
    images = []
    labels = []

    # Loop through each class directory
    for class_index, class_name in enumerate(class_names):
        class_directory = os.path.join(dataset_directory, class_name)

        # Loop through each image file in the class directory
        for image_filename in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_filename)

            image_array = load_image(image_path, image_size)

            images.append(image_array)
            labels.append(class_index)  # Assign label based on class index

    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def handle_data(images, labels):
    print('Handle data..')

    # Split the data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    return train_images, test_images, train_labels, test_labels


def create_model(image_size, num_channels, num_classes, weights_file, train=False):
    print('Make model..')

    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, num_channels)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    if train:
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print('Model complied.')

    else:
        model.load_weights(weights_file)
        print('Model weights loaded.')

    return model


def train_model(model, train_images, train_labels, batch_size, epochs, weights_file):
    print('Train model..')

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

    # Save the model weights
    model.save_weights(weights_file)
    print('Model saved.')
    print()


def test_model(model, test_images, test_labels):
    print('Test model..')

    # Calculate accuracy
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


def load_image(path, image_size):

    img = Image.open(path)
    img = img.resize((image_size, image_size))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values

    return img_array


def predict(model, new_images):
    # print('Predict images..')

    # Make predictions
    predictions = model.predict(new_images)

    # The predictions will be in the form of probabilities for each class
    # You can convert them to class labels if needed
    predicted_labels = np.argmax(predictions, axis=1)

    # print(labels[predicted_labels[0]], end=" ")
    # print("Predicted labels:", labels[predicted_labels[0]])

    print(f'{np.round(predictions, 2)}')

    return predicted_labels

def predict_image(model, image, chunk_size, num_classes):

    print('Predict image..')

    import cv2

    # Load the larger image
    image_height, image_width, _ = image.shape

    # Initialize empty lists to store patch locations and predictions
    patch_locations = []
    patch_predictions = []
    patch_predictions_dict = {i: [] for i in range(0, num_classes)}

    # Iterate through the image with the sliding window
    for y in range(0, image_height, chunk_size):
        print(f'\r{y}/{image_height}', end="")

        for x in range(0, image_width, chunk_size):
            patch = image[y:y + chunk_size, x:x + chunk_size, :]

            # Preprocess the patch and predict using the model
            patch = cv2.resize(patch, (chunk_size, chunk_size))
            patch = np.expand_dims(patch, axis=0)
            patch = patch / 255.0

            prediction = model.predict(patch)
            predicted_class = np.argmax(prediction)

            # patch_locations.append((x, y))
            # patch_predictions.append(predicted_class)

            location = (x, y)
            patch_predictions_dict[predicted_class].append(location)

    print()

    # # Print patch locations and predictions
    # for location, prediction in zip(patch_locations, patch_predictions):
    #     print(f"Patch at location {location} predicted as class {prediction}")

    import json
    # save dictionary to person_data.pkl file
    json_filename = 'chunk_predict.json'
    with open(json_filename, 'w') as json_file:
        json.dump(patch_predictions_dict, json_file)
        print('dictionary saved successfully to file')

def load_json(file_name):
    import json

    # Read JSON data from a file
    with open(file_name, 'r') as file:
        data_dict = json.load(file)
    return data_dict


if __name__ == '__main__':

    train = False  # TRUE WHEN TRAINING MODEL!!

    dataset_directory = 'classified/chunks'

    # Input and labels
    image_size = 25
    num_channels = 3
    num_classes = 5

    # Train the model
    batch_size = 32
    epochs = 25
    weights_file = 'cnn_model_weights.h5'

    model = create_model(image_size, num_channels, num_classes, weights_file, train=train)

    if train:
        images, labels = load_data(dataset_directory, image_size)
        train_images, test_images, train_labels, test_labels = handle_data(images, labels)
        train_model(model, train_images, train_labels, batch_size, epochs, weights_file)
        test_model(model, test_images, test_labels)

    # Evaluate
    new_images = []
    gnd_truth = []
    for i in range(10):
        label = random.choice(label_classes)
        gnd_truth.append(label_classes.index(label))
        image_array = load_image(f'classified/chunks/{label}/0.png', 25)
        new_images.append(image_array)
    new_images = np.array(new_images)
    predictions = predict(model, new_images)
    predictions = np.round(predictions, 2)
    gnd_truth = np.array(gnd_truth)
    print(f'{gnd_truth} gnd_truth')
    print(f'{predictions} predictions')
    correct = 1 * (gnd_truth == predictions)
    print(f'{correct} {int(np.sum(correct) / len(correct)) * 100}%')
