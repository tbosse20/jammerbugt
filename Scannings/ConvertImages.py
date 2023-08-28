import os
import PyPDF2
import cv2
import numpy as np


def extract_image_from_pdf(pdf_path, output_folder, file_num):
    pdf = PyPDF2.PdfReader(pdf_path)
    image_count = 0

    for page_num, page in enumerate(pdf.pages):
        xObject = page['/Resources']['/XObject'].get_object()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                image_count += 1
                if image_count % 2 == 0: continue
                img = xObject[obj].get_data()
                img_file_path = os.path.join(output_folder, f"image_{file_num}.png")
                with open(img_file_path, 'wb') as f:
                    f.write(img)


def extract_images_from_pdfs(pdf_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_num, pdf_filename in enumerate(os.listdir(pdf_folder)):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            extract_image_from_pdf(pdf_path, output_folder, file_num)


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


def split_image(image, chunk_size, width, height):
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = image[y:y + chunk_size, x:x + chunk_size]
            chunks.append(chunk)

    return chunks


def split_images(input_folder, chunk_size, output_folder):

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(input_folder):
        if not image_file.endswith(".png"): continue
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        classification = image_file.split('_')[0]
        tmp_output_folder = os.path.join(output_folder, classification)

        # Create output folder if it doesn't exist
        if not os.path.exists(tmp_output_folder):
            os.makedirs(tmp_output_folder)

        # Split the image into chunks
        chunks = split_image(image, chunk_size, 1000, 1025)
        count_chunks = 0
        for i, chunk in enumerate(chunks):

            has_dimensions = np.any(np.array(chunk.shape) == 0)
            if has_dimensions: continue

            red = np.array([0, 0, 255])
            red_mask = cv2.inRange(chunk, red, red)
            if np.any(red_mask): continue

            # chunk_filename = f"chunk_{image_file.split('_')[1][:-4]}_{i}.png"
            chunk_filename = f"{count_chunks}.png"
            count_chunks += 1
            chunk_path = os.path.join(tmp_output_folder, chunk_filename)
            cv2.imwrite(chunk_path, chunk)

def expand(image, divider):
    height, width, _ = image.shape

    border_size_x = divider - (width % divider)
    border_size_y = divider - (height % divider)

    expanded_width = width + border_size_x
    expanded_height = height + border_size_y
    expanded_image = np.zeros(
        (expanded_height, expanded_width, 3),
        dtype=np.uint8)

    # Calculate the position to place the original image
    x_pos = int(border_size_x / 2)
    y_pos = int(border_size_y / 2)

    # Place the original image on the canvas
    expanded_image[y_pos:y_pos + height, x_pos:x_pos + width] = image

    return expanded_image


def rotate_image(image, angle):
    height, width = image.shape[:2]  # Get image dimensions
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # Calculate the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))  # Perform the rotation

    return rotated_image

def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    coords = np.column_stack(np.where(gray > 0))  # Find the coordinates of non-black pixels
    x, y, w, h = cv2.boundingRect(coords)  # Calculate the bounding box of the non-black region
    image = image[x:x + w, y:y + h]  # Crop the image
    return image


def crop_red(image):
    red = np.array([0, 0, 255])
    mask = cv2.inRange(image, red, red)
    coords = np.column_stack(np.where(mask == 0))  # Find the coordinates of red pixels

    if len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)  # Calculate the bounding box of the red region
        cropped_image = image[x:x + w, y:y + h]  # Crop the image based on the bounding box
        return cropped_image
    else:
        return None  # No red region found

# Main script
if __name__ == "__main__":
    pdf_folder = "Files"
    output_folder = "training_data"
    chunk_size = 25

    print(f'Processing files in size {chunk_size}..')

    # extract_images_from_pdfs(pdf_folder, output_folder)
    split_images(output_folder, chunk_size)

    print(f'Complete.')
