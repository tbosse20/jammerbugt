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

def separate_colors(anotated_img_path):

    cover_color = [0, 0, 255]
    output_folder = ConvertImages.handle_sub_folder(anotated_img_path, 'images')

    image = cv2.imread(anotated_img_path)

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

    super_folder = 'classified'
    # path_no = 'em070.006_binned_jsf-ch12_SS_No Interp.TIF'
    anotated_img_path = os.path.join(super_folder, 'em070.006_binned_jsf-ch12_SS_Interp.TIF')
    chunk_size = 25

    separate_colors(anotated_img_path)
    ConvertImages.split_images(super_folder, chunk_size, save=True)
