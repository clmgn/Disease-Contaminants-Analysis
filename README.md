# Disease-Contaminants-Analysis

This project was developed as part of an engineering internship, where the goal was to find a way of visualizing potential links between the presence of air pollutants and actual diseases. It is dedicated to hospital centers and health specialits.

The final results of this project are displayed on an interactive map from OpenStreetMap (OSM). On this map are added the Catalunyans 'comarques' using the file 'map_comarques.html'. Then, the source codes can calculate and show on the map the contaminants and patients location on the map, considering the contaminants' concentration (downloaded online from the script), the disease of interest and the patients data. All the project is based upon a metadata file containing patients' information. It can only work if this file has the requested format, with the right amount and names of columns.

## Repository Structure
<pre/> Disease-Contaminants-Analysis/
├── data-csv/ → useful data so that the code can run properly (CSV format)
├── src/ → source code (data processing & visualization)
├── comarques-compressed.geojson → geographic boundaries (GeoJSON), also necessary to have
├── influenced_fake_files.py → script that can be used to make a fake metadata.csv file with the right format
├── map_comarques.html → necessary file to add the comarques boundaries on the map
├── .gitignore
└── README.md </pre>

## Interface explanation

The app can be tested by launching the 'front.py' file from the 'src/' directory. Once it is open, follow these instructions to fill out the main window :

1. Select a metadata file from your local browser (you can create a fake one with 'influenced_fake_files.py', or if you already have one, make sure that the inside format is correct).
2. Enter the dates on which you want to study the effect of the contaminants. It must be at the format "YYYY-MM-DD".
3. Validate first step.
4. Choose the disease you want to visualize on the map. Validate Step 2.
5. Select a contaminant. They are listed so that the first one on the list has the highest probability of being found at the places as the selected disease.
6. Choose the kind of map you want to display.
7. Validate and display the map.
