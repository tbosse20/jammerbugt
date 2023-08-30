import random
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os, cv2, json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models

from Scannings import ConvertImages


def load_data(dataset_directory):
    print('Loading data..')

    # List of class names (subdirectory names)
    class_names = os.listdir(dataset_directory)

    # Initialize lists to store chunks and corresponding labels
    chunks = list()
    labels = list()
    # Loop through each class directory
    for label_idx, label_name in enumerate(class_names):
        label_dir = os.path.join(dataset_directory, label_name)

        # Loop through each image file in the class directory
        for chunk_filename in os.listdir(label_dir):
            chunk_path = os.path.join(label_dir, chunk_filename)
            chunk_array = load_chunk(chunk_path)
            chunks.append(chunk_array)
            labels.append(label_idx)

    # Shuffle data
    combined = list(zip(chunks, labels))
    random.shuffle(combined)
    chunks, labels = zip(*combined)

    # Convert the lists to numpy arrays
    chunks, labels = np.array(chunks), np.array(labels)

    return chunks, labels


def handle_data(images, labels):
    print('Handle data..')

    # Split the data into training and testing sets
    train_chunks, test_chunks, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    return train_chunks, test_chunks, train_labels, test_labels


def create_model(image_size, num_channels, num_classes, weights_file, train=False):
    print('Make model..')

    # Define the CNN model
    model = models.Sequential([
        # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, num_channels)),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(128, (3, 3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.GlobalAveragePooling2D(),
        # layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, activation='softmax')

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, num_channels)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')

        # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(chunk_size, chunk_size, num_channels)),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(128, (3, 3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.GlobalAveragePooling2D(),
        # layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, activation='softmax')
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


def train_model(model, train_chunks, train_labels, batch_size, epochs, weights_file):
    print('Train model..')

    model.fit(train_chunks, train_labels, batch_size=batch_size, epochs=epochs)

    # Save the model weights
    model.save_weights(weights_file)
    print('Model saved.')
    print()


def test_model(model, test_chunks, test_labels):
    print('Test model..')

    # Calculate accuracy
    loss, accuracy = model.evaluate(test_chunks, test_labels)



def load_chunk(path):
    chunk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    chunk_array = np.array(chunk) / 255.0  # Normalize pixel values
    return chunk_array


def predict(model, new_chunks):
    # print('Predict images..')

    # Make predictions
    predictions = model.predict(new_chunks)

    # The predictions will be in the form of probabilities for each class
    # You can convert them to class labels if needed
    predicted_labels = np.argmax(predictions, axis=1)

    # print(labels[predicted_labels[0]], end=" ")
    # print("Predicted labels:", labels[predicted_labels[0]])

    print(f'{np.round(predictions, 2)}')

    return predicted_labels


def predict_image(model, image, chunk_size, num_classes, json_filename):
    print('Predicting images..')

    # Load the larger image
    image_height, image_width = image.shape

    # Initialize empty lists to store patch locations and predictions
    patch_predictions_dict = {i: [] for i in range(0, num_classes)}

    # Iterate through the image with the sliding window
    for y in range(0, image_height, chunk_size):
        print(f'\r{y}/{image_height}', end="")

        for x in range(0, image_width, chunk_size):
            chunk = image[y:y + chunk_size, x:x + chunk_size]

            # Preprocess the chunk and predict using the model
            chunk = cv2.resize(chunk, (chunk_size, chunk_size))
            chunk = np.expand_dims(chunk, axis=0)
            chunk = chunk / 255.0

            prediction = model.predict(chunk, verbose=None)
            predicted_class = np.argmax(prediction)

            location = (x, y)
            patch_predictions_dict[predicted_class].append(location)

    print()

    # save dictionary to person_data.pkl file
    with open(json_filename, 'w') as json_file:
        json.dump(patch_predictions_dict, json_file)
        print('dictionary saved successfully to file')

    return patch_predictions_dict

def load_json(file_name):
    import json

    print('Loading json..')

    # Read JSON data from a file
    with open(file_name, 'r') as file:
        data_dict = json.load(file)
    return data_dict


def kfolding(chunks, labels, chunk_size, num_channels):
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras import callbacks

    # Number of folds
    num_folds = 5

    # Initialize the KFold cross-validator
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Lists to store accuracy values for each fold
    fold_accuracies = []

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(chunks, labels)):
        print(f"Fold {fold + 1}/{num_folds}")

        chunks_train, chunks_val = chunks[train_idx], chunks[val_idx]
        labels_train, labels_val = labels[train_idx], labels[val_idx]

        # Create your model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(chunk_size, chunk_size, num_channels)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Define the ModelCheckpoint callback to save weights
        checkpoint_path = f'classified/weights_fold_{fold + 1}.h5'
        checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)

        # Train the model
        model.fit(
            chunks_train, labels_train,
            epochs=10, batch_size=32,
            validation_data=(chunks_val, labels_val),
            callbacks=[checkpoint],
            verbose=None
        )

        # Evaluate the model on the validation data
        _, accuracy = model.evaluate(chunks_val, labels_val)
        print(f"Validation accuracy: {accuracy}")

if __name__ == '__main__':

    train = True  # TRUE WHEN TRAINING MODEL!!

    main_folder = 'classified'
    dataset_directory = os.path.join(main_folder, 'output', 'chunks')

    # Input and labels
    chunk_size = 25
    num_channels = 1
    label_classes = ["brown", "dark blue", "kaki", "light blue", "yellow"]

    # Train the model
    batch_size = 32
    epochs = 50
    weights_file = 'classified/cnn_model_weights.h5'

    num_classes = len(label_classes)
    model = create_model(chunk_size, num_channels, num_classes, weights_file, train=train)

    if train:
        chunks, labels = load_data(dataset_directory)
        kfolding(chunks, labels, chunk_size, num_channels)
        exit()
        train_chunks, test_chunks, train_labels, test_labels = handle_data(chunks, labels)
        train_model(model, train_chunks, train_labels, batch_size, epochs, weights_file)
        test_model(model, test_chunks, test_labels)

    # Evaluate
    new_images = []
    gnd_truth = []
    for i in range(10):
        label = random.choice(label_classes)
        gnd_truth.append(label_classes.index(label))
        image_array = load_chunk(f'classified/output/chunks/{label}/0.png')
        new_images.append(image_array)
    new_images = np.array(new_images)
    predictions = predict(model, new_images)
    predictions = np.round(predictions, 2)
    gnd_truth = np.array(gnd_truth)
    print(f'{gnd_truth} gnd_truth')
    print(f'{predictions} predictions')
    correct = 1 * (gnd_truth == predictions)
    print(f'{correct} {int(np.sum(correct) / len(correct) * 100)}%')
