import rasterio, random
import numpy as np
from matplotlib import pyplot as plt
import download_points, data_types
import xml.etree.ElementTree as ET
from PIL import Image

''' Export file using QGIS
Load from WMS https://docs.qgis.org/3.28/en/docs/training_manual/online_resources/wms.html
Export https://maps.cga.harvard.edu/qgis_2/wkshop/export_GeoTiff.php
'''



def genereate_random_points(src, num_samples=10):
    '''
    Generate random points to check if "sample_point" works
    '''
    # Get the image dimensions
    width = src.width
    height = src.height

    # Generate random sample points within the image bounds
    sample_points = []
    for _ in range(num_samples):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        sample_points.append((x, y))
    return sample_points


def sample_point(point, src):
    '''
    Sample points on map compared with "Soil_types"

    :param point: Point coordinates
    :param src: Tif map src to read
    :return: Type of soil
    '''

    x, y = point

    if x <= 0: return
    if x >= src.width: return
    if y <= 0: return
    if y >= src.height: return

    rgb = [1, 2, 3]
    rgb_values = list()
    for c in rgb:
        rgb_value = src.read(c, window=((y, y + 1), (x, x + 1)))[0][0]
        rgb_values.append(rgb_value)
    soil = None
    for tmp_soil in data_types.Seabeds:
        if tmp_soil.value != rgb_values: continue
        soil = tmp_soil.name
        break
    # print(f"Point ({x}, {y}): Value = {soil} ({rgb_values})")
    return soil


def map_value(value, start1, stop1, start2, stop2, within_bounds=True):
    '''
    Map values between two start and stops
    (Made with ChatGPT, and p5 method)
    '''
    mapped_value = np.interp(value, (start1, stop1), (start2, stop2))
    if within_bounds:
        mapped_value = np.clip(mapped_value, start2, stop2)
    return mapped_value

def plot_map(image_data, plots):
    '''
    Display the map with scatter
    '''
    plt.imshow(image_data)
    for key, values in plots.items():
        plt.scatter(values['plot_x'], values['plot_y'], color=values['color'], s=5)
    plt.axis('off')
    # plt.show()
    plt.savefig(f'presentation/seabed_map_with_plots.png')

def update_coordinates(point, geus_map:dict, src):
    coordinates = point.find('coordinates').text
    coordinates = list(map(float, coordinates.split(',')))  # [float(i) for i in coordinates]
    x = map_value(coordinates[0], geus_map['x'][0], geus_map['x'][1], 0, src.width, within_bounds=False)
    y = map_value(coordinates[1], geus_map['y'][0], geus_map['y'][1], src.height, 0, within_bounds=False)
    return x, y
def crop_tif_avoiding_black(image_path: str):
    '''
    Crop tif file to only have seabed data bounding box
    Made with ChatGPT
    :return: Cropped tif file
    '''
    image = Image.open(image_path)  # Open the image using PIL
    grayscale_image = image.convert('L')  # Convert the image to grayscale
    bbox = grayscale_image.getbbox()  # Get the bounding box coordinates of the non-black region
    cropped_image = image.crop(bbox)  # Crop the image using the bounding box coordinates
    cropped_image.save('cropped_' + image_path)
    print(f'tif file cropped!')
    return cropped_image
def sample_points_on_map(tif_path, points_file):

    # Manual set coordinates from map located on data.gues.dk
    # TODO: Maybe not precise
    geus_map = {
        "x": [142319.03722847934, 975509.4100721639],  # [WEST, EAST]
        "y": [6027326.68020619, 6458218.862079183],  # [SOUTH, NORTH]
    }

    download_points.update_check(points_file)

    plots = {
        'marta_video': {'color': 'blue', 'plot_x': list(), 'plot_y': list()},
        'marta_images': {'color': 'red', 'plot_x': list(), 'plot_y': list()}
    }

    crop_tif_avoiding_black(tif_path)
    with rasterio.open('cropped_' + tif_path) as src:
        image_data = src.read([1, 2, 3])
        image_data = image_data.transpose(1, 2, 0)

        tree_points = ET.parse(points_file)  # Get XML points file
        root_points = tree_points.getroot()
        # print(f'width: {src.width}, height: {src.height}')

        for media in data_types.Medias:
            media_points = root_points.find(media.name, data_types.namespace)
            points = media_points.findall('point', data_types.namespace)

            for i, point in enumerate(points):
                # if i >= 50: break

                # Update coordinates according to GEUS map offset
                x, y = update_coordinates(point, geus_map, src)

                # Append coordinates to plot
                plots[media.name]['plot_x'].append(x)
                plots[media.name]['plot_y'].append(y)

                # Sample point according to seabed
                seabed = sample_point([x, y], src)
                if seabed is None: seabed = data_types.Seabeds.UNLOCATED.name

                # Append or update assigned seabed to point element
                seabed_feature = point.find('seabed')
                if seabed_feature is None:
                    seabed_feature = ET.Element('seabed')
                    point.append(seabed_feature)
                seabed_feature.text = seabed

        tree_points.write(points_file)  # Update XML point file

        plot_map(image_data, plots)
def visualize_seabed_classification(points_file):
    '''
    Visualize number of seabed classifications for "videos" and "images"
    (With ChatGPT)
    '''

    tree_points = ET.parse(points_file)  # Get XML points file
    root_points = tree_points.getroot()

    # Loop medias
    for i, media in enumerate(data_types.Medias):
        media_points = root_points.find(media.name, data_types.namespace)
        points = media_points.findall('point', data_types.namespace)

        # Count each point
        sub_element_count = {seabed.name: 0 for seabed in data_types.Seabeds}
        for point in points:
            seabed_text = point.find('seabed').text
            sub_element_count[seabed_text] += 1

        # Remove unlocated attribute if empty
        unlocated = data_types.Seabeds.UNLOCATED.name
        if sub_element_count[unlocated] <= 0:
            sub_element_count.pop(unlocated)

        labels = list(sub_element_count.keys())
        counts = list(sub_element_count.values())

        plt.bar(labels, counts)
        plt.xlabel('Seabed')
        plt.ylabel('Count')
        plt.title(f'Seabed data "{media.name[6:]}"')
        plt.xticks(rotation=90)
        plt.grid(color='gray', linestyle='dashed', alpha=0.5)
        plt.tight_layout()  # Adjust the layout to prevent labels from going out of bounds
        # plt.show()
        plt.savefig(f'presentation/seabed_data_{media.name[6:]}.png')

if __name__ == "__main__":

    tif_path = 'seabed_sediments_map.tif'  # Map obtained from QGIS (https://we.tl/t-hCZY6itJPa)
    points_file = 'points.xml'  # XML file of points to sample (obtained with "download_points.py")

    sample_points_on_map(tif_path, points_file)
    # visualize_seabed_classification(points_file)
