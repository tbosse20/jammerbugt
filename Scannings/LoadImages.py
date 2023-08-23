import os
import cv2

def loadImages(image_folder, bnw=False):
    images = []
    for filename in os.listdir(image_folder):
        if not filename.endswith(".png"): continue

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if bnw:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        images.append(img)

    return images

def split_image(image, chunk_size):
    height, width, _ = image.shape
    chunks = []

    for y in range(0, 1025, chunk_size):
        for x in range(0, 1000, chunk_size):
            chunk = image[y:y + chunk_size, x:x + chunk_size]
            chunks.append(chunk)

    return chunks

def split_images(input_folder, chunk_size):

    output_folder = "output_chunks"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(input_folder):
        if not image_file.endswith(".png"): continue
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        # Split the image into chunks
        chunks = split_image(image, chunk_size)

        for i, chunk in enumerate(chunks):
            chunk_filename = f"chunk_{image_file.split('_')[1][:-4]}_{i}.png"
            chunk_path = os.path.join(output_folder, chunk_filename)
            cv2.imwrite(chunk_path, chunk)

# Main script
if __name__ == "__main__":
    pass