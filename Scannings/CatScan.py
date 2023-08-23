import math
from collections import defaultdict

from sklearn.cluster import KMeans
import LoadImages
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg

def preprocess_and_cluster_images(image_folder, num_clusters=5):

    images = LoadImages.loadImages(image_folder)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(images)

    cluster_labels = kmeans.predict(images)
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    for label, count in zip(unique_labels, label_counts):
        print(f"Cluster {label}: {count} data points")

    image_filenames = [
        filename for filename in os.listdir(image_folder)
    ]

    # Assuming 'image_filenames' is a list of image file paths
    # and 'cluster_labels' is a list of cluster labels assigned to each image
    image_data = list(zip(image_filenames, cluster_labels))

    # Creating a dictionary to group images by cluster label
    cluster_images = defaultdict(list)
    for image_filename, label in image_data:
        cluster_images[label].append(image_filename)
    print(cluster_images)

    # Show first image from each cluster
    if False:
        fig = plt.figure(figsize=(8, 8))
        columns = int(math.sqrt(len(unique_labels)))
        rows = int(math.sqrt(len(unique_labels)))
        ax = []  # ax enables access to manipulate each of subplots
        # Print out images under each cluster category
        for label in range(len(unique_labels)):
            image_filename = cluster_images[label][0]
            image_filepath = os.path.join(image_folder, image_filename)
            img = mpimg.imread(image_filepath)

            ax.append(fig.add_subplot(rows, columns, label + 1))
            ax[label].set_title(f'Label: {label}')  # set title
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    # Show first ten images from each cluster
    if True:
        fig = plt.figure(figsize=(9, 9))
        columns = 20
        rows = len(unique_labels)
        ax = []  # ax enables access to manipulate each of subplots

        # Print out images under each cluster category
        for label in range(len(unique_labels)):
            for i in range(columns):
                image_filename = cluster_images[label][i]
                image_filepath = os.path.join(image_folder, image_filename)
                img = mpimg.imread(image_filepath)
                ax.append(fig.add_subplot(rows, columns, (columns * label) + i + 1))
                plt.imshow(img)
                plt.axis('off')
        plt.show()


    # Show images in same cluster
    if False:
        show_label = 0

        fig = plt.figure(figsize=(8, 8))
        columns = int(math.sqrt(len(cluster_images[show_label])))
        rows = int(math.sqrt(len(cluster_images[show_label])))
        ax = []  # ax enables access to manipulate each of subplots
        # Print out images under each cluster category
        for i, image_filename in enumerate(cluster_images[show_label]):
            if i >= columns * rows - 1: break
            if i >= 200: break

            image_filepath = os.path.join(image_folder, image_filename)
            img = mpimg.imread(image_filepath)

            ax.append(fig.add_subplot(rows, columns, i + 1))
            plt.imshow(img)
            plt.axis('off')

        plt.show()

    return kmeans


if __name__ == "__main__":
    num_clusters = 25
    output_folder = "output_chunks"

    kmeans = preprocess_and_cluster_images(output_folder, num_clusters)

    print("Cluster labels:", np.array(kmeans.labels_))
