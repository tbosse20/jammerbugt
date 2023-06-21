import os
import pandas as pd
import requests


def download_files(
        excel_file: str,  # Column title for counting words
        file_column: str,
        contents: str,
        file_type: str,
        limit=None,
):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    file_ids = df[file_column]
    total_files = file_ids.count()
    print(f'Found {total_files} {contents} files')

    for i, id in enumerate(file_ids):

        if i >= limit: break

        print(f'{i}/{total_files} Downloading {contents} {id=} ..', end=" ")
        url = f'https://data.geus.dk/geusmapmore/marta/info_marta_getfile.jsp?iContents={contents}&iID={id}'
        response = requests.get(url)
        print("Status", end="=")

        if response.status_code != 200:
            print("Failed!")
            continue

        folder_path = f'downloads/{contents}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = str(id) + '.' + file_type
        full_path = folder_path + file_name
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Success.")

    print()


download_files(
    excel_file='marta_video.xlsx',
    file_column='id',
    contents='video',
    file_type='mp4',
    limit=2,
)

download_files(
    excel_file='marta_images.xlsx',
    file_column='id',
    contents='images',
    file_type='jpg',
    limit=5,
)
