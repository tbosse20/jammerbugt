import OpenTIF2
import CNN
import numpy as np
import random
import cv2
import os

if __name__ == '__main__':

    # Load image
    folder = 'SideScans'
    angle = -30

    # Input and labels
    chunk_size = 25
    num_channels = 3
    num_classes = 5

    # CNN
    weights_file = 'cnn_model_weights.h5'

    # Draw
    alpha = 0.5

    # -----------------------------------

    images = OpenTIF2.load_images(folder, angle, chunk_size)
    image = images[0]
    image = image[:1000, :]

    if False:
        model = CNN.create_model(chunk_size, num_channels, num_classes, weights_file)
        CNN.predict_image(model, image, chunk_size, num_classes)

    chunk_predict = CNN.load_json('chunk_predict.json')
    print(chunk_predict)

    def draw_chunk(image, chunk_predict):
        output_image = np.zeros_like(image)  # Create a blank canvas for marking clusters

        cluster_colors = [
            [30, 255, 255],  # 1b
            [30, 255, 192],  # 2a
            [0, 255, 128],  # 2b
            [90, 255, 255],  # 3
            [120, 255, 255],  # 4
        ]

        for label, chunks in chunk_predict.items():
            for location in chunks:

                x, y = location

                color = cluster_colors[int(label)]
                output_image[y:y + chunk_size, x:x + chunk_size] = color

        out_img = np.zeros(image.shape, dtype=image.dtype)
        out_img[:, :, :] = (alpha * image[:, :, :]) + ((1 - alpha) * output_image[:, :, :])

        # Save the output image
        filename = f'predicted_image.tif'
        cv2.imwrite(filename, out_img)
        cv2.imshow(filename, out_img)
        cv2.waitKey(0)

    draw_chunk(image, chunk_predict)