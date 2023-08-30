import copy, os, cv2, random, ConvertImages, math, ConvertImages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict


def load_images(super_folder, chunk_size, angle=0, save=False):
    print(f'Load images..')

    folder_name = ConvertImages.handle_sub_folder(super_folder, 'output/images')

    # Load the TIF image
    files = os.listdir(super_folder)
    images = list()
    for i, file_path in enumerate(files):
        check_tif_file = file_path.endswith(".tif")
        check_png_file = file_path.endswith(".png")
        if not (check_tif_file or check_png_file): continue

        path = os.path.join(super_folder, file_path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image

        black_pixels = np.where(image[:, :] == 255)
        image[black_pixels] = [0]  # set those pixels to white

        image = ConvertImages.rotate_image(image, angle)
        image = ConvertImages.crop_image(image)
        image = ConvertImages.expand(image, chunk_size, 1)
        images.append(image)

        # Save the output image
        if not save: continue
        file_name = file_path.split(".tif")[0].split("\\")[-1]
        new_file_path = f'{file_name}.png'
        new_file_path = os.path.join(folder_name, new_file_path)
        cv2.imwrite(new_file_path, image)

    return images


def proc_images(images, chunk_size):
    print(f"Process images..")

    chunks = []
    original_chunks = []

    for image_num, image in enumerate(images):

        original = copy.deepcopy(image)

        # Invert background
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(thresholded)

        # Edge detect
        image = cv2.GaussianBlur(image, (3, 3), 2)
        image = cv2.Canny(image=image, threshold1=150, threshold2=200)  # Canny Edge Detection

        # Threshold
        # _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        # image = cv2.GaussianBlur(image, (5, 5), 10)

        image[inverted > 0] = 255  # Invert background

        # Divide the image into chunks
        height, width = image.shape
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):

                chunk = original[y:y + chunk_size, x:x + chunk_size]
                check_chunk_size = np.any(np.array(chunk.shape[:2]) != chunk_size)
                if check_chunk_size: continue

                # Check red areas
                red_area_detected = False
                red = np.array([0, 0, 255])
                red_mask = cv2.inRange(chunk, red, red)
                if np.any(red_mask):
                    chunk[:, :] = [0, 0, 255]
                    red_area_detected = True

                original_chunks.append(chunk)

                chunk = image[y:y + chunk_size, x:x + chunk_size]
                if red_area_detected: chunk[:, :] = [100]
                chunks.append(chunk.flatten())

    print(f'{len(chunks)=}')

    return chunks, original_chunks


def cluster(chunks):
    print('Clustering..')
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=0,
    )
    kmeans.fit(chunks)
    cluster_labels = kmeans.predict(chunks)
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    chunk_idxs = [i for i in range(len(chunks))]
    chunk_data = list(zip(chunk_idxs, cluster_labels))

    for label, count in zip(unique_labels, label_counts):
        print(f"Cluster {label}: {count} data points")

    # Creating a dictionary to group images by cluster label
    cluster_images = defaultdict(list)
    for chunk_idx, label in chunk_data:
        cluster_images[label].append(chunk_idx)

    return cluster_labels, cluster_images, unique_labels


def draw_clusters(super_folder, images, chunks, cluster_labels):
    print('Drawing clusters..')

    # Define colors for each cluster
    cluster_colors = [
        (random.randint(0, 255),
         random.randint(0, 255),
         random.randint(0, 255))
        for _ in range(num_clusters)
    ]

    output_folder = ConvertImages.handle_sub_folder(super_folder, 'drawn')

    tot_chunk_count = 0
    for i, original in enumerate(images):

        output_image = np.zeros_like(original)  # Create a blank canvas for marking clusters
        # output_image = ConvertImages.expand(output_image, chunk_size, 1)
        height, width, _ = original.shape

        # Mark clusters on the output image
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):

                chunk_idx = y // chunk_size * (width // chunk_size) + x // chunk_size
                chunk_idx = tot_chunk_count + chunk_idx

                if chunk_idx >= len(chunks): break

                cluster_label = cluster_labels[chunk_idx]
                color = cluster_colors[cluster_label]
                output_image[y:y + chunk_size, x:x + chunk_size] = color

        tot_chunk_count += int((height * width) / math.pow(chunk_size, 2))

        out_img = np.zeros(original.shape, dtype=original.dtype)
        out_img[:, :, :] = (alpha * original[:, :, :]) + ((1 - alpha) * output_image[:, :, :])

        # Save the output image
        filename = f'drawn_image_{i}.png'
        file_path = os.path.join(output_folder, filename)
        cv2.imwrite(file_path, out_img)


def show_clusters(unique_labels, cluster_images, original_chunks):
    fig = plt.figure(figsize=(15, 8))
    columns = 10
    rows = len(unique_labels)
    ax = []  # ax enables access to manipulate each of subplots

    # Print out images under each cluster category
    for label in range(len(unique_labels)):
        for i in range(columns):
            if i >= len(cluster_images[label]): break
            ax.append(fig.add_subplot(rows, columns, (columns * label) + i + 1))
            plt.axis('off')
            # print(cluster_images)
            random_idx = random.choice(cluster_images[label])
            # print(len(cluster_images[label]), random_idx)
            # chunk_idx = cluster_images[label][random_idx]
            chunk = original_chunks[random_idx]
            plt.imshow(chunk)
    plt.show()


if __name__ == "__main__":
    num_clusters = 7
    # path = 'em064.001_binned_jsf-ch34/em064.001_binned_jsf-ch34.tif'
    folder = 'SideScans'
    angle = -30
    chunk_size = 25
    alpha = 0.5

    images = load_images(folder, chunk_size, angle, save=True)
    images = ConvertImages.red_backgrounds(images)
    chunks, original_chunks = proc_images(images, chunk_size)
    cluster_labels, cluster_images, unique_labels = cluster(chunks)
    draw_clusters(folder, images, chunks, cluster_labels)
    show_clusters(unique_labels, cluster_images, original_chunks)
