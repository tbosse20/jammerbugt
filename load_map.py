from enum import Enum
import rasterio
import random
import numpy as np
from matplotlib import pyplot as plt
import handle_points

''' Export file using QGIS
Load from WMS https://docs.qgis.org/3.28/en/docs/training_manual/online_resources/wms.html
Export https://maps.cga.harvard.edu/qgis_2/wkshop/export_GeoTiff.php
'''


class Soil_types(Enum):
    DYND = [56, 168, 0]
    DYNET_SAND = [152, 230, 0]
    SAND = [255, 255, 0]
    GRUS = [255, 152, 0]
    MORAENE = [120, 50, 0]
    LER = [168, 112, 0]
    GRUNDFJELD = [217, 52, 52]
    UNKNOWN = [0, 0, 0]
    UNLOCATED = None


def genereate_random_points(src, num_samples=10):
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
    x, y = point

    if x <= 0: return
    if x >= src.width: return
    if y <= 0: return
    if y >= src.height: return

    rgb_values = list()
    for c in rgb:
        rgb_value = src.read(c, window=((y, y + 1), (x, x + 1)))[0][0]
        rgb_values.append(rgb_value)
    soil = None
    for tmp_soil in Soil_types:
        if tmp_soil.value != rgb_values: continue
        soil = tmp_soil.name
        break
    # print(f"Point ({x}, {y}): Value = {soil} ({rgb_values})")
    return soil


if __name__ == "__main__":

    tif_path = 'high3.tif'
    rgb = [1, 2, 3]

    # TODO: Manual (UPDATE!)
    geus_map_x = [120591.75430574606, 1000073.6239711933]  # [LEFT, RIGHT]
    geus_map_y = [6004745.402733848, 6458218.862079183]  # [BOTTOM, TOP]


    def map_value(value, start1, stop1, start2, stop2, within_bounds=True):
        '''
        ChatGPT
        :param value:
        :param start1:
        :param stop1:
        :param start2:
        :param stop2:
        :param within_bounds:
        :return:
        '''
        mapped_value = np.interp(value, (start1, stop1), (start2, stop2))
        if within_bounds:
            mapped_value = np.clip(mapped_value, start2, stop2)
        return mapped_value


    with rasterio.open(tif_path) as src:
        image_data = src.read(rgb)
        image_data = image_data.transpose(1, 2, 0)

        typenames = ['marta_images', 'marta_video']
        points_file = 'points.xml'
        plots = {
            'marta_video': {
                'color': 'blue',
                'plot_x': list(),
                'plot_y': list(),
                'soil_types': {soil_type.name: [] for soil_type in Soil_types}
            },'marta_images': {
                'color': 'red',
                'plot_x': list(),
                'plot_y': list(),
                'soil_types': {soil_type.name: [] for soil_type in Soil_types}
            }
        }

        sample_points = handle_points.load_points_to_dict(points_file, typenames)
        print(f'width: {src.width}, height: {src.height}')

        for typename in typenames:
            image_samples = sample_points[typename]['point']

            for i, sample in enumerate(image_samples):
                # if i >= 50: break

                coordinates = sample['coordinates']
                coordinates = list(map(float, coordinates.split(',')))  # [float(i) for i in coordinates]
                x = map_value(coordinates[0], geus_map_x[0], geus_map_x[1], 0, src.width, within_bounds=False)
                y = map_value(coordinates[1], geus_map_y[0], geus_map_y[1], src.height, 0, within_bounds=False)
                plots[typename]['plot_x'].append(x)
                plots[typename]['plot_y'].append(y)

                soil = sample_point([x, y], src)
                if soil is None: soil = Soil_types.UNLOCATED.name
                id = sample['id']
                plots[typename]['soil_types'][soil].append(id)

        # Display the image
        plt.imshow(image_data)
        for key, values in plots.items():
            plt.scatter(values['plot_x'], values['plot_y'], color=values['color'])
        plt.axis('off')
        plt.show()
