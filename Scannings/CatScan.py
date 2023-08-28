import math, os
from collections import defaultdict
from sklearn.cluster import KMeans
import ConvertImages
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def show_clusters(columns, unique_labels, cluster_images, image_folder):
    fig = plt.figure(figsize=(18, 9))
    rows = len(unique_labels)
    ax = []  # ax enables access to manipulate each of subplots

    # Print out images under each cluster category
    for label in range(len(unique_labels)):
        for i in range(columns):
            image_filename = cluster_images[label][i]
            image_filepath = os.path.join(image_folder + '_org', image_filename)
            img = mpimg.imread(image_filepath)
            ax.append(fig.add_subplot(rows, columns, (columns * label) + i + 1))
            plt.imshow(img)
            plt.axis('off')
    plt.show()


def show_cluster(show_label, columns, cluster_images, image_folder):
    fig = plt.figure(figsize=(8, 8))
    rows = int(math.sqrt(len(cluster_images[show_label])))
    ax = []  # ax enables access to manipulate each of subplots
    # Print out images under each cluster category
    for i, image_filename in enumerate(cluster_images[show_label]):
        if i >= columns * rows - 1: break
        if i >= 200: break

        image_filepath = os.path.join(image_folder + '_org', image_filename)
        img = mpimg.imread(image_filepath)

        ax.append(fig.add_subplot(rows, columns, i + 1))
        plt.imshow(img)
        plt.axis('off')

    plt.show()


def preprocess_and_cluster_images(image_folder, num_clusters=5):
    images = ConvertImages.loadImages(image_folder)

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
    # print(cluster_images)

    # Show first ten images from each cluster
    show_clusters(10, unique_labels, cluster_images, image_folder)

    # Show images in same cluster
    show_cluster(0, 10, cluster_images, image_folder)

    return kmeans


if __name__ == "__main__":
    num_clusters = 25
    output_folder = "output_chunks"

    kmeans = preprocess_and_cluster_images(output_folder, num_clusters)

    print("Cluster labels:", np.array(kmeans.labels_))
