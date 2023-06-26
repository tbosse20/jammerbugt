from download_points import *
from sample_points_on_map import *

''' 
Download and handle XML file with points, and sample then upon the .tif map
'''

points_file = 'points.xml'  # XML file of points to sample (obtained with "download_points")

# Download points
include = {
    'id': './/ns1:id',
    'coordinates': './/gml:Point/gml:coordinates',
    # 'lokalitet': './/ns1:lokalitet',
}

download_files()
filter_files(points_file)

# Sample points on map
tif_path = 'seabed_sediments_map.tif'  # Map obtained from QGIS (https://we.tl/t-xFYj7HObw8)

# Manual set coordinates from map located on data.gues.dk
# TODO: Maybe not precise
geus_map = {
    "x": [120591.75430574606, 1000073.6239711933],  # [LEFT, RIGHT]
    "y": [6004745.402733848, 6458218.862079183],  # [BOTTOM, TOP]
}

sample_points_on_map(tif_path, geus_map, points_file)
visualize_seabed_classification(points_file)