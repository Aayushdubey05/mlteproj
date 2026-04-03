import pandas as pd
import os
import numpy
from pathlib import Path


data_path = './datasets'
all_dataframes = []

columns_to_drop = ['Common in Region', 'Language Availability', 'Patient_ID', 'Admission_Date', 'Discharge_Date','Patient_State','Year_of_Admission',
                   'Length_of_Stay','Readmission','Satisfaction','Insurance_Claimed','Total_Cost','State_Name','Cholesterol_Level','Triglyceride_Level','LDL_Level','HDL_Level','Systolic_BP','Diastolic_BP',
                   'Annual_Income', 'Health_Insurance']


for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        df = pd.read_csv(file_path,low_memory=False)

        # Drop 'X' column if it exists
        cols_to_drop = [cols for cols in columns_to_drop if cols in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped {cols_to_drop} column from {filename}")

        df["source_file"] = filename

        all_dataframes.append(df)
        print(f"loaded {filename}: {len(df)} rows")

combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

combined_df.to_csv(f"{data_path}/inputFile.csv", index=False)

print(f"The total number of the rows in the final input file is {len(combined_df)}")
