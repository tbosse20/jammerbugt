
# Kig Under Vandet - Aalborg University
## _Gues data training_

## Description
- Download data from GEUS MARTA in form of points, video, images, and seabed sediments. 
- Classify each points according to seabed sediments.
- Process each point with downloaded media.
Data source: [Geus Marta data](
https://data.geus.dk/geusmap/?mapname=marta#baslay=baseMapDa&optlay=&extent=19081.47838710714,5990066.985780745,1043081.4783871071,6481066.985780745)

## Warning
- Do not upload the _"seabed_sediments_map.tif"_ to Github, as it is too big.
  
## Run
- Process media files:
  - Write _"call_method"_ to handle each media.
  - Remove _"limit"_ parameter.
  - Run _"download_medias_xml.py"_
- Classify seabed points:
  - Download and insert the _"seabed_sediments_map.tif"_ file. Link found in "Resources".
  - Run _"classify_points"_.

## Presentation
Points on map with seabed🔴
<p float="left">
  <img src="presentation/seabed_map_with_plots.png" width="400" />
</p>

Graphs classified data
<p float="left">
  <img src="presentation/seabed_data_video.png" width="400" />
  <img src="presentation/seabed_data_images.png" width="400" /> 
</p>

## Features
1. Download "image" and "video" XML files, and filter points with point id and coordinates. _("download_points.py")_
2. Download all images or videos from Geus Marta Map into a 'downloads' file from the XML point file, and process. _("download_medias.xml.py")_
3. Sample points on map according to seabed sediments _("sample_points_on_map")_

## Resources
- __points.xml__ (existing of point id, coordinates, and seabed classification)
- __marta_video.xml__
- __marta_image.xml__
- __seabed_sediments_map.tif__ (Download link: [https://we.tl/t-hCZY6itJPa](https://we.tl/t-hCZY6itJPa))

### Contributors
- Tonko Bossen, Aalborg University

## License - MIT
