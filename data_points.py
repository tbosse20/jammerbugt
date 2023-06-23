import requests
import xml.etree.ElementTree as ET
import xmltodict as xmltodict


def download_files(typenames: list, points_file: str, maxfeatures: int = 1):
    ''' Download files and save them as xml '''
    for typename in typenames:
        print(f'Processing: "{typename}"', end=". ")
        url = f'https://data.geus.dk/geusmap/ows/25832.jsp?nocache=nocache&whoami=178.157.255.197&typename={typename}&service=WFS&version=1.0.0&mapname=marta&maxfeatures={maxfeatures}&outputformat=gml2&request=GetFeature'
        response = requests.get(url)
        if response.status_code == 200:
            file_path = typename + '.xml'
            with open(file_path, "wb") as file:
                file.write(response.content)
            print("Success!")
        else:
            raise "Failed to download the file."


def filter_files(typenames: list, include: dict):
    namespace = {
        'gml': 'http://www.opengis.net/gml',
        'ms': 'http://example.com/ms',
        'ns0': 'http://www.opengis.net/gml',
        'ns1': 'http://mapserver.gis.umn.edu/mapserver'
    }

    # Create a new root element for the output XML
    output_root = ET.Element("root")

    for typename in typenames:

        # Parse the XML file
        xml_file_path = typename + '.xml'
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        if len(root) >= maxfeatures: raise "Found more features! Increase 'maxfeatures'"

        featureMember_elements = root.findall('.//gml:featureMember', namespace)
        print(f'Points in {typename}: {len(featureMember_elements)}')

        typename_root = ET.Element(typename)
        for i, featureMember_element in enumerate(featureMember_elements):
            if i >= maxfeatures: break

            point = ET.Element('point')

            for key, find in include.items():
                sub_element = featureMember_element.find(find, namespace)
                if sub_element is None:
                    raise f'"{key}" not found'
                sub_element.tag = key
                point.append(sub_element)

            typename_root.append(point)

        output_root.append(typename_root)

    # Save the XML file locally
    tree = ET.ElementTree(output_root)
    tree.write(points_file, encoding="utf-8", xml_declaration=True)
    # print('XML Element:', ET.tostring(tree, encoding='unicode'))

    return output_root


def load_points_to_dict(points_file: str, typenames: list):
    ''' Load XML points to dict '''
    with open(points_file, 'r') as file:
        xml_data = file.read()

    data_dict = xmltodict.parse(xml_data)['root']

    # Access and manipulate the dictionary data as needed
    for typename, content in data_dict.items():
        if not typename in typenames: continue
        points = content['point']
        print(f'{typename}: {len(points)}')


if __name__ == "__main__":
    typenames = ['marta_images', 'marta_video']
    include = {
        'id': './/ns1:id',
        'coordinates': './/gml:Point/gml:coordinates',
        'lokalitet': './/ns1:lokalitet',
    }
    points_file = 'points.xml'
    maxfeatures = 7000

    download_files(typenames, points_file, maxfeatures)
    filter_files(typenames, include)
    load_points_to_dict(points_file, typenames)
