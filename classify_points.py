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

sample_points_on_map(tif_path, points_file)
visualize_seabed_classification(points_file)