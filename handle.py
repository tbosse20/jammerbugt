import xml.etree.ElementTree as ET

namespace = {
    'gml': 'http://www.opengis.net/gml',
    'ms': 'http://example.com/ms'
}

file_path = "images.xml"  # Replace with the actual file path

tree = ET.parse(file_path)
root = tree.getroot()

point_elements = root.findall('.//gml:Point', namespace)
for point_element in point_elements:
    coordinates_element = point_element.find('gml:coordinates', namespace)
    if coordinates_element is not None:
        coordinates = coordinates_element.text
        print(coordinates)

print(f'Points found: {len(root)}')
