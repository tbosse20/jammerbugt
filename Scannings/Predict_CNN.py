import numpy as np
import Clustering, CNN, random, cv2, os
from Scannings import ConvertImages


def draw_chunk(image, chunk_predict, i):

    print('Drawing chunks..')

    output_image = np.zeros_like(image)  # Create a blank canvas for marking clusters

    output_folder = ConvertImages.handle_sub_folder(main_folder, 'predicted')

    cluster_colors = [
        [30, 255, 255],  # yellow: Substrate 1b
        [30, 255, 192],  # kaki: Substrate 2a
        [0, 255, 128],  # brown: Substrate 2b
        [90, 255, 255],  # light blue: Substrate 3
        [120, 255, 255],  # dark blue: Substrate 4
    ]

    for label, chunks in chunk_predict.items():
        for location in chunks:
            x, y = location

            color = cluster_colors[int(label)]
            output_image[y:y + chunk_size, x:x + chunk_size] = color

    out_img = np.zeros(image.shape, dtype=image.dtype)
    out_img[:, :, :] = (alpha * image[:, :, :]) + ((1 - alpha) * output_image[:, :, :])

    # Save the output image
    filename = f'image{i}.png'
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, out_img)
    # cv2.imshow(filename, out_img)
    # cv2.waitKey(0)


if __name__ == '__main__':

    predict = True

    # Load image
    folder = 'SideScans'
    angle = -30

    # Input and labels
    chunk_size = 25
    num_channels = 1
    num_classes = 5

    # CNN
    main_folder = 'classified'
    weights_file = 'cnn_model_weights.h5'
    weights_file = 'weights_fold_5.h5'
    weights_path = os.path.join(main_folder, weights_file)
    json_filename = os.path.join(main_folder, 'chunk_predict.json')

    # Draw
    alpha = 0.5

    # -----------------------------------

    # folder = os.path.join(main_folder)
    images = Clustering.load_images(folder, chunk_size, angle, save=False)

    # image = images[0]
    # images = [image]
    # images[0] = images[0][:1000, :]

    for i, image in enumerate(images):
        if predict:
            model = CNN.create_model(chunk_size, num_channels, num_classes, weights_path)
            chunk_predict = CNN.predict_image(model, image, chunk_size, num_classes, json_filename)
        else:
            chunk_predict = CNN.load_json(json_filename)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        draw_chunk(image, chunk_predict, i)