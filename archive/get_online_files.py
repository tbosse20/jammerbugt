import random

''' Get information about the files in data.geus.dk '''

''' !!!!!!!!!!!!!!!!!!!!! '''
''' !!! OUTDATED FILE !!! '''
''' !!!!!!!!!!!!!!!!!!!!! '''
''' Data obtained easier using 'process_media.py' '''

urls = {
    "nord": "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=50&J=50&CRS=EPSG%3A25832&STYLES=&WIDTH=10&HEIGHT=10&BBOX=513876.9290123458,6399293.595679009,520384.2592592594,6405800.9259259235&X=5&Y=5&FEATURE_COUNT=100",
    "south": "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=50&J=50&CRS=EPSG%3A25832&STYLES=&WIDTH=10&HEIGHT=10&BBOX=567712.54530223,6070367.380401232,572050.7654668392,6074705.6005658405&X=5&Y=5&FEATURE_COUNT=100",
    "east": "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=50&J=50&CRS=EPSG%3A25832&STYLES=&WIDTH=10&HEIGHT=10&BBOX=713670.460387701,6110030.47946293,714955.8589549927,6111315.878030221&X=5&Y=5&FEATURE_COUNT=100",
    "west": "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=50&J=50&CRS=EPSG%3A25832&STYLES=&WIDTH=10&HEIGHT=10&BBOX=348972.001198391,6306747.491781033,349828.93357658543,6307604.424159228&X=5&Y=5&FEATURE_COUNT=100",
    "west2": "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=50&J=50&CRS=EPSG%3A25832&STYLES=&WIDTH=10&HEIGHT=10&BBOX=393006.44716699334,6306323.267831432,393863.37954518775,6307180.200209627&X=5&Y=5&FEATURE_COUNT=100"
}


def split(string):
    start = "BBOX="
    end = "&X"
    start_idx = string.index(start)
    end_idx = string.index(end)
    string = string[start_idx + 5:end_idx]
    data = string.split(",")
    data = [int(float(i)) for i in data]
    return data


data = {}
for dir, url in urls.items():
    coor = split(url)
    data[dir] = coor
print(data)

import requests

url = "https://data.geus.dk/geusmap/map.jsp?&0=0&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeatureInfo&FORMAT=image%2Fpng&TRANSPARENT=true&QUERY_LAYERS=marta_video&LAYERS=marta_video&MAPNAME=marta&EPSG=EPSG%3A25832&FILTER=&INFO_FORMAT=text%2Fhtml&I=100&J=100&CRS=EPSG%3A25832&STYLES=&WIDTH=50&HEIGHT=50&BBOX="
url_end = "&X=50&Y=50&FEATURE_COUNT=100"
for i in range(20):
    d = []
    for i in range(4):
        random_key = random.choice(list(data.keys()))
        w = data[random_key][i]
        d.append(str(w))
    coords = ','.join(d)
    print(coords, end=", ")
    f_url = url + coords + url_end
    x = requests.get(f_url)
    print(x.text.count("<tr>"))
