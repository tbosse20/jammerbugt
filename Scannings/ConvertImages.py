import copy
import os
import PyPDF2
import cv2
import numpy as np

def red_backgrounds(images):
    new_images = []
    for image in images:
        mask = copy.deepcopy(image)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image[np.where(mask == 0)] = [0, 0, 255]
        new_images.append(image)
    return new_images


def check_folder_existence(folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)


def handle_sub_folder(super_folder, sub_folder):
    super_folder = super_folder.split("\\")[0]
    sub_folder = os.path.join(super_folder, sub_folder)
    check_folder_existence(sub_folder)

    return sub_folder


def extract_image_from_pdf(pdf_path):
    folder_name = handle_sub_folder(pdf_path, 'output/images')

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
        if not filename.endswith(".tif"): continue

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if bnw: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        images.append(img)

    return images

def split_image(image, chunk_size, class_folder=None, save=False, flatten=True):
    height, width, _ = image.shape
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = image[y:y + chunk_size, x:x + chunk_size]

            # Check dimensions
            has_dimensions = np.any(np.array(chunk.shape) == 0)
            if has_dimensions: return

            # Check correct dimensions
            check_chunk_size = np.any(np.array(chunk.shape[:2]) != chunk_size)
            if check_chunk_size: continue

            # Check red areas
            red = np.array([0, 0, 255])
            red_mask = cv2.inRange(chunk, red, red)
            if np.any(red_mask): continue

            chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
            chunks.append(chunk)

            if not save: continue
            if class_folder is None: continue
            chunk_filename = f"{len(chunks) - 1}.png"
            chunk_path = os.path.join(class_folder, chunk_filename)
            cv2.imwrite(chunk_path, chunk)

            # print(chunk)
            # print(chunk.shape)
            # cv2.imshow("awd", chunk)
            # cv2.waitKey(0)

    return chunks


def split_images(super_folder, chunk_size, save=False):
    print('Splitting images into chunks..')

    folder_name = handle_sub_folder(super_folder, 'output/chunks')
    image_folder = os.path.join(super_folder, 'output', 'images')

    # Iterate each file in folder
    chunks = []
    images = os.listdir(image_folder)
    for img_num, image_file in enumerate(images):
        check_tif_file = image_file.endswith(".tif")
        check_png_file = image_file.endswith(".png")
        if not (check_tif_file or check_png_file): continue

        print(f'Image: {img_num}/{len(images)}')

        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        classification = image_file.split('_')[0]
        class_folder = os.path.join(folder_name, classification)
        check_folder_existence(class_folder)

        chunks += split_image(image, chunk_size, class_folder, save)  # Split the image into chunks

    print(f'{len(chunks)=}')
    return chunks


def expand(image, divider, size):
    height, width = image.shape

    border_size_x = (divider - (width % divider)) * size
    border_size_y = (divider - (height % divider)) * size

    expanded_width = width + border_size_x
    expanded_height = height + border_size_y
    expanded_image = np.zeros(
        (expanded_height, expanded_width),
        dtype=np.uint8)

    # Calculate the position to place the original image
    x_pos = int(border_size_x / 2)
    y_pos = int(border_size_y / 2)

    # Place the original image on the canvas
    expanded_image[
    y_pos:y_pos + height,
    x_pos:x_pos + width
    ] = image

    return expanded_image


def rotate_image(image, angle):
    height, width = image.shape[:2]  # Get image dimensions
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # Calculate the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))  # Perform the rotation
    return rotated_image


def crop_image(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    coords = np.column_stack(np.where(image > 0))  # Find the coordinates of non-black pixels
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
