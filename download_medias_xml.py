import os, requests
import xml.etree.ElementTree as ET
import data_types
import download_points

def request_media_url(url):
    '''
    Request url to download media
    '''

    response = requests.get(url)
    # Status response check
    if response.status_code != 200:
        print("Failed!")
        return None
    return response


def save_file(response, folder_path, full_path):
    '''
    Save file locally with given paths
    '''

    # Check folder existence
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save file locally in 'downloads' file
    with open(full_path, 'wb') as file:
        file.write(response.content)
    print(f"Success", end=". ")


def download_medias_from_points(
        content: data_types.Medias,     # Media type to download
        points_file,                    # Path to XML point file
        limit=None,                     # Limit number of downloads
        call_method=None,               # Call method after each download
):
    '''
    Download files from separate media from Contents
    '''

    check_points_exist(points_file)

    # Parse the XML file
    tree = ET.parse(points_file)  # Get XML file
    root = tree.getroot()
    xml_content = root.find(content.name)

    points = xml_content.findall('point')  # Separate each element
    print(f'Points in {content.name}: {len(points)}')

    for i, point in enumerate(points):

        if i >= limit: break  # Break at limit reached

        point_id = point.find('id').text
        media_type = content.name[len('marta_'):]
        print(f'{i}/{len(points)} Downloading {media_type} {point_id=} ..', end=" ")
        url = f'https://data.geus.dk/geusmapmore/marta/info_marta_getfile.jsp?iContents={media_type}&iID={point_id}'
        response = request_media_url(url)

        folder_path = f'downloads/{content.name}/'
        file_name = str(point_id) + '.' + content.name
        full_path = folder_path + file_name
        save_file(response, folder_path, full_path)

        # Call method
        if call_method:
            call_method(full_path)
            print(f'Method called', end=". ")
            os.remove(full_path) # Delete file

        print()
def check_points_exist(points_file):
    # Check XML points existence
    if os.path.exists(points_file):
        return

    download_points.download_files()
    download_points.filter_files(points_file)

def download_medias_from_contents(
        points_file,                # Path to XML point file
        limit=None,                 # Limit number of downloads
        call_method=None,           # Call method after each download
):
    '''
    Download all images or videos from Geus Marta Map into a 'downloads' file.
    Callable method between each iteration, with a deletion of the file afterwards
    '''

    check_points_exist(points_file)

    # Loop "videos" and "images"
    for media in data_types.Medias:
        download_medias_from_points(media, points_file, limit, call_method)


def call_method(path: str) -> None:
    '''
    Method to call after each download
    :param path: Path to downloaded file
    '''
    pass


if __name__ == "__main__":
    points_file = 'points.xml'

    download_medias_from_contents(points_file, limit=2, call_method=call_method)
    download_medias_from_points(data_types.Medias.marta_video, points_file, limit=2, call_method=call_method)
