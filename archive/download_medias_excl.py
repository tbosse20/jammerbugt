import os, requests
import pandas as pd
from enum import Enum


class Contents(Enum):
    VIDEO = 'video'
    IMAGES = 'images'

def download_media_url(url, folder_path, full_path):
    response = requests.get(url)

    # Status response check
    if response.status_code != 200:
        print("Failed!")
        return

    # Check folder existence
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save file locally in 'downloads' file
    with open(full_path, 'wb') as file:
        file.write(response.content)
    print(f"Success", end=". ")

def download_medias(
        excel_file: str,        # Name of Excel file to load from
        file_column: str,       # Column to extract data from
        contents: Contents,     # Video or image data
        file_type: str,         # File type to save to
        limit=None,             # Limit number of downloads
        callable_method=None,   # Call method after each download
        del_after: bool = None  # Delete after method called
) -> None:
    '''
    Download all images or videos from Geus Marta Map into a 'downloads' file.
    Callable method between each iteration, with a deletion of the file afterwards

    Steps to download Excel file:
    1. Open: https://data.geus.dk/geusmap/?mapname=marta#baslay=baseMapDa&optlay=&extent=-400000,5500000,1000000,6000000
    2. "Images and video"
    3. "Seabed images" or "Seabed video"
    4. "Layer Extent"
    5. "Export"
    6. Extract .zip file
    7. Open .dbf file in Excel
    '''

    df = pd.read_excel(excel_file)  # Load the Excel file into a DataFrame

    file_ids = df[file_column]
    total_files = file_ids.count()
    print(f'Found {total_files} {contents.value} files')

    # Iterate all rows in Excel file
    for i, id in enumerate(file_ids):

        if i >= limit: break  # Break at limit reached

        # Obtain url
        url = f'https://data.geus.dk/geusmapmore/marta/info_marta_getfile.jsp?iContents={contents.value}&iID={id}'
        print(f'{i}/{total_files} Downloading {contents.value} {id=} ..', end=" ")

        folder_path = f'downloads/{contents.value}/'
        file_name = str(id) + '.' + file_type
        full_path = folder_path + file_name

        response = requests.get(url)

        # Status response check
        if response.status_code != 200:
            print("Failed!")
            return

        # Check folder existence
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save file locally in 'downloads' file
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Success", end=". ")

        # Call method
        if callable_method:
            callable_method(full_path)
            print(f'Method called', end=". ")

        # Delete file
        if del_after:
            os.remove(full_path)
            print(f'DELETED', end=". ")

        print()

    print()


def callable_method(path: str) -> None:
    '''
    Method to call after each download
    :param path: Path to downloaded file
    '''
    pass


if __name__ == "__main__":
    download_medias(
        excel_file='marta_video.xlsx',
        file_column='id',
        contents=Contents.VIDEO,
        file_type='mp4',
        limit=2,
        callable_method=callable_method,
        del_after=True
    )

    download_medias(
        excel_file='marta_images.xlsx',
        file_column='id',
        contents=Contents.IMAGES,
        file_type='jpg',
        limit=5,
        del_after=True
    )
