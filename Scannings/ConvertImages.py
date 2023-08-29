import os
import PyPDF2
import cv2
import numpy as np

def check_folder_existence(folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

def handle_sub_folder(super_folder, sub_folder):

    super_folder = super_folder.split("\\")[0]
    sub_folder = os.path.join(super_folder, 'output', sub_folder)
    check_folder_existence(sub_folder)

    return sub_folder

def extract_image_from_pdf(pdf_path):

    folder_name = handle_sub_folder(pdf_path, 'images')

    pdf = PyPDF2.PdfReader(pdf_path)

    image_count = 0
    for page_num, page in enumerate(pdf.pages):
        xObject = page['/Resources']['/XObject'].get_object()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                image_count += 1
                if image_count % 2 == 0: continue  # Discard measurements
                img = xObject[obj].get_data()
                pdf_name = pdf_path.split(".")[0].split("\\")[-1]
                img_name = f"{pdf_name}.png"
                img_file_path = os.path.join(folder_name, img_name)
                with open(img_file_path, 'wb') as f:
                    f.write(img)


def extract_images_from_pdfs(pdf_folder):
    
    print('Extracting images from pdfs..')

    for file_num, pdf_filename in enumerate(os.listdir(pdf_folder)):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            extract_image_from_pdf(pdf_path)


def loadImages(image_folder, bnw=False):

    check_folder_existence(image_folder)

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


def split_images(super_folder, chunk_size, save=False):

    print('Splitting images into chunks..')

    folder_name = handle_sub_folder(super_folder, 'chunks')
    image_folder = os.path.join(super_folder, 'output', 'images')

    # Iterate each file in folder
    chunks = []
    for img_num, image_file in enumerate(os.listdir(image_folder)):
        print(f'\r{img_num}')
        if not image_file.endswith(".png"): continue
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        classification = image_file.split('_')[0]
        class_folder = os.path.join(folder_name, classification)

        # Create classification folder
        if not os.path.exists(class_folder): os.makedirs(class_folder)

        # Split the image into chunks
        chunks = split_image(image, chunk_size, 1000, 1025)
        count_chunks = 0
        for i, chunk in enumerate(chunks):

            # Check dimensions
            has_dimensions = np.any(np.array(chunk.shape) == 0)
            if has_dimensions: continue

            # Check red areas
            red = np.array([0, 0, 255])
            red_mask = cv2.inRange(chunk, red, red)
            if np.any(red_mask): continue

            if not save: continue
            # chunk_filename = f"chunk_{image_file.split('_')[1][:-4]}_{i}.png"
            chunk_filename = f"{count_chunks}.png"
            chunk_path = os.path.join(class_folder, chunk_filename)
            cv2.imwrite(chunk_path, chunk)
            count_chunks += 1
            chunks.append(chunk)
    return chunks

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
    pdf_folder = "S230815_2 scans"
    chunk_size = 25

    print(f'Processing files in size {chunk_size}..')

    extract_images_from_pdfs(pdf_folder)
    split_images(pdf_folder, chunk_size, save=True)

    print(f'Complete.')
