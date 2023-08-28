import cv2, ConvertImages, os, math, CatScan, copy
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = 'em064.001_binned_jsf-ch34\em064.001_binned_jsf-ch34.tif'
    output_folder = "chucnks"
    chunk_size = 25
    angle = -30
    num_clusters = 4

    image = Image.open(path)  # Open the image using PIL
    image = ConvertImages.rotate_image(image, angle)
    image = image[:, 1110:1410]

    org = copy.deepcopy(image)

    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    image = cv2.GaussianBlur(image, (5, 5), 10)

    chunks = ConvertImages.split_image(image, chunk_size, 300, 4050)
    chunks_org = ConvertImages.split_image(org, chunk_size, 300, 4050)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder + '_org'):
        os.makedirs(output_folder + '_org')

    fig = plt.figure(figsize=(8, 8))
    columns = int(math.sqrt(len(chunks)))
    rows = int(math.sqrt(len(chunks)))
    ax = []  # ax enables access to manipulate each of subplots
    # Print out images under each cluster category

    print(f'{len(chunks)=}')
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i}.png"
        chunk_path = os.path.join(output_folder, chunk_filename)
        cv2.imwrite(chunk_path, chunk)
        chunk_path = os.path.join(output_folder + '_org', chunk_filename)
        cv2.imwrite(chunk_path, chunks_org[i])

        # ax.append(fig.add_subplot(rows, columns, i + 1))
        # plt.imshow(chunk)
        # plt.axis('off')
        # if i >= len(chunks) - 10: break

    kmeans = CatScan.preprocess_and_cluster_images(output_folder, num_clusters)

    print('')
