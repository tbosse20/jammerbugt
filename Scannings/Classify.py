import numpy as np
import copy, cv2, os
import ConvertImages

substracts = {
    "red": [0, 255, 255],
    "yellow": [30, 255, 255],  # 1b
    "kaki": [30, 255, 192],  # 2a
    "brown": [0, 255, 128],  # 2b
    "light blue": [90, 255, 255],  # 3
    "dark blue": [120, 255, 255],  # 4
}

def separate_colors(path, output_folder):

    cover_color = [0, 0, 255]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(path)

    _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    edge = copy.deepcopy(threshold)
    edge = cv2.GaussianBlur(edge, (5, 5), 0)
    edge = cv2.Canny(edge, threshold1=175, threshold2=200)  # Canny Edge Detection

    # kernel = np.ones((25, 25), np.uint8)
    # proc_edge = cv2.dilate(edge, kernel, 5)
    # kernel = np.ones((25, 25), np.uint8)
    # proc_edge = cv2.erode(proc_edge, kernel, 5)
    # test = copy.deepcopy(image)
    # test[np.where(proc_edge == 0)] = [0]
    # cv2.imshow(f'Color Cont', test)


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_area = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, (name, value) in enumerate(substracts.items()):
        color_mask = cv2.inRange(hsv_image, np.array(value), np.array(value))

        kernel = np.ones((25, 25), np.uint8)
        color_mask = cv2.dilate(color_mask, kernel, 5)
        kernel = np.ones((10, 10), np.uint8)
        color_mask = cv2.erode(color_mask, kernel, 5)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for con_num, contour in enumerate(contours):

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            white = (255, 255, 255)
            cv2.drawContours(mask, contours, con_num, white, cv2.FILLED)  # You can change the color and thickness as needed

            # Redraw outline
            if name != "red":
                black = (0, 0, 0)
                cv2.drawContours(mask, contours, -1, black, 30)

            filtered = copy.deepcopy(image)
            filtered[np.where(mask == 0)] = cover_color
            filtered[np.where(red_area == 255)] = cover_color
            filtered = ConvertImages.crop_red(filtered)
            # print(filtered.shape)
            # cv2.imshow("file", filtered)
            cv2.waitKey(0)

            if name == "red":
                red_area[np.where(mask == 255)] = [255]
                continue

            file = f'{name}_{con_num}'
            # cv2.namedWindow(file, cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty(file, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            # cv2.imshow(file, filtered)

            file += '.png'
            save_path = os.path.join(output_folder, file)
            cv2.imwrite(save_path, filtered)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    path = 'em070.006_binned_jsf-ch12_SS_Interp.TIF'
    # path_no = 'em070.006_binned_jsf-ch12_SS_No Interp.TIF'
    classified_folder = 'classified'
    output_folder = os.path.join(classified_folder, 'images')
    chunk_folder = os.path.join(classified_folder, 'chunks')
    chunk_size = 25

    # Create output folder if it doesn't exist
    if not os.path.exists(classified_folder):
        os.makedirs(classified_folder)

    separate_colors(path, output_folder)
    ConvertImages.split_images(output_folder, chunk_size, chunk_folder)
