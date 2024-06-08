import pandas as pd

metadata_path = 'Clothing_Dataset_small/styles.csv'  
cleaned_metadata_path = 'cleaned_metadata.csv'

# clean csv file
expected_fields = 10  

with open(metadata_path, 'r', encoding='utf-8') as infile, open(cleaned_metadata_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if len(line.split(',')) == expected_fields:
            outfile.write(line)

# load data
metadata = pd.read_csv(cleaned_metadata_path)
print(metadata.head())
