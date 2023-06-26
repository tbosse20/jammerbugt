import os

import requests
import xml.etree.ElementTree as ET
import xmltodict as xmltodict
import data_types

def download_files(max_elements: int = 7000):
    '''
    Download image and video files and save them as xml
    "marta_images.xml" and "marta_videos.xml"
    '''

    # Loop "videos" and "images"
    for media in data_types.Medias:
        print(f'Processing: "{media.name}"', end=". ")

        # Obtain url
        url = f'https://data.geus.dk/geusmap/ows/25832.jsp?nocache=nocache&whoami=178.157.255.197&typename={media.name}&service=WFS&version=1.0.0&mapname=marta&maxfeatures={max_elements}&outputformat=gml2&request=GetFeature'
        response = requests.get(url)

        # Check response status
        if response.status_code != 200:
            raise "Failed to download the file."

        # Save XML file locally
        file_path = media.name + '.xml'
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("Success!")


def filter_files(points_file: str, max_elements: int=7000) -> ET.Element:
    '''
    Filter XML file only passing included features

    :return: New saved XML file with included elements
    '''

    # Feature to include
    include = {
        'id': './/ns1:id',
        'coordinates': './/gml:Point/gml:coordinates',
    }

    output_root = ET.Element("root")  # Create a new root element for the output XML

    # Loop "videos" and "images"
    for media in data_types.Medias:

        # Parse the XML file
        xml_file_path = media.name + '.xml'
        tree = ET.parse(xml_file_path)  # Get XML file
        root = tree.getroot()

        # Features seeking not high enough
        if len(root) >= max_elements:
            raise "Found more features! Increase 'max_elements'"

        # Separate each element
        featureMember_elements = root.findall('.//gml:featureMember', data_types.namespace)
        print(f'Points in {media.name}: {len(featureMember_elements)}')

        # Make new "video" or "image" sub element
        typename_root = ET.Element(media.name)
        for i, featureMember_element in enumerate(featureMember_elements):
            if i >= max_elements: break  # Break at element limit

            point = ET.Element('point')  # Make new "point" element

            # Loop each feature to include
            for key, find in include.items():
                # Find feature in element
                sub_element = featureMember_element.find(find, data_types.namespace)
                if sub_element is None:
                    raise f'"{key}" not found'
                sub_element.tag = key  # Update key to desired
                point.append(sub_element)

            typename_root.append(point)

        output_root.append(typename_root)

    # Save the XML file locally
    tree = ET.ElementTree(output_root)
    tree.write(points_file, encoding="utf-8", xml_declaration=True)
    # print('XML Element:', ET.tostring(tree, encoding='unicode'))

    return output_root


def load_points_to_dict(points_file: str):
    ''' Load XML points to dict '''
    with open(points_file, 'r') as file:
        xml_data = file.read()

    data_dict = xmltodict.parse(xml_data)['root']

    # Access and manipulate the dictionary data as needed
    for media_name, feature in data_dict.items():
        if not data_types.Medias.has_member_key(media_name): continue
        points = feature['point']
        print(f'{media_name}: {len(points)}')

    return data_dict

def update_check(file_path):

    # Check points existence
    if os.path.exists(file_path):
        update_points = input(f'Points already downloaded. Update? (y/n): ')
        if update_points != 'y':
            print('Aborted..')
            exit()

if __name__ == "__main__":
    points_file = 'points.xml'

    update_check(points_file)

    download_files()
    filter_files(points_file)
