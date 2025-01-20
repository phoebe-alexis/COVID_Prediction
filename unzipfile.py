import zipfile
import os

# Path to the zip file
zip_file_path = "/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/cb_2018_us_state_500k.zip"

# Destination folder for extracted files
extract_to = "shapefile_data"

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Shapefile folder extracted to: {os.path.abspath(extract_to)}")
