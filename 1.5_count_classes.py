import pandas as pd

# load data
file_path = 'cleaned_metadata.csv'
df = pd.read_csv(file_path)

# show column that will be used
print(df.columns)

# Counting masterCategory from csv (cleaned_metadata.csv)
# can be changed to see each Category (column) 
class_column = 'masterCategory'

# Counts unique classes
unique_classes = df[class_column].value_counts()

print("Unique classes and count:")
print(unique_classes)
