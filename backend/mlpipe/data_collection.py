import pandas as pd
import os
import numpy 
from pathlib import Path


data_path = './datasets'
all_dataframes = []

for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        df = pd.read_csv(file_path, low_memory=False)
        df["source_file"] = filename

        all_dataframes.append(df)
        print(f"loaded {filename}: {len(df)} rows")

combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

combined_df.to_csv(f"{data_path}/inputFile.csv", index=False)

print(f"The total number of the rows in the final input file is {len(combined_df)}") 



