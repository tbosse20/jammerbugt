# Kig Under Vandet - Aalborg University
### _Gues data training_

### Contributors
- Tonko Bossen, Aalborg University 

### Description
Download data from GEUS MARTA in form of points, video, images, and seabed sediments. 
Classify each points according to seabed sediments for machine learning training.
Data source: [Geus Marta data](
https://data.geus.dk/geusmap/?mapname=marta#baslay=baseMapDa&optlay=&extent=19081.47838710714,5990066.985780745,1043081.4783871071,6481066.985780745)

## Run
- Download and insert the _"seabed_sediments_map.tif"_ file. Link found in "Resources".
- Run "main.py".

## Features
- Download "image" and "video" XML files, and filter points with id, coordinates, and download location. _("download_points.py")_
- Download all images or videos from Geus Marta Map into a 'downloads' file from the XML point file. _("download_medias.xml.py")_
- Sample points on map according to seabed sediments _("sample_points_on_map")_

## Resources
- __points.xml__
- __marta_video.xml__
- __marta_image.xml__
- __seabed_sediments_map.tif__ (https://we.tl/t-xFYj7HObw8)

## License - MIT
