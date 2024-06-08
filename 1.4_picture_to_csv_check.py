import os
import pandas as pd

# Path to csv file and images
images_path = 'Clothing_Dataset_small/images'
cleaned_metadata_path = 'cleaned_metadata.csv'  

# load cleaned csv
metadata = pd.read_csv(cleaned_metadata_path)

# id will be used as string and makes sure of it
metadata['id'] = metadata['id'].astype(str)

# Checked if all Pictures have an entry in the csv
missing_files = []
for img_id in metadata['id']:
    img_path = os.path.join(images_path, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        missing_files.append(img_id)

# show missing Data
print(f"Count of missing Files: {len(missing_files)}")


if missing_files:
    print("Missing Files:", missing_files[:10])  # show first 10 missing files 
else:
    print("All Files are available and are correct.")
