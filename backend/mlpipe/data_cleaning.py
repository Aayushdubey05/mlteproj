import pandas as pd
import os
import re
from pathlib import Path

input_data = "./datasets/inputFile.csv"

def clean_column_names(df):
    """Clean column names: lowercase, spaces to underscores, remove special chars."""
    cleaned = []
    seen = {}

    for col in df.columns:
        new_col = re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_'))
        new_col = new_col.strip('_')

        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        cleaned.append(new_col)

    df.columns = cleaned
    return df

def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower, upper)
    return df


df = pd.read_csv(input_data, low_memory=False)

# Standardize column names
df = clean_column_names(df)
print("Columns:", list(df.columns))

print(len(df))
df['patient_id'] = range(1, len(df)+1)
print(df['patient_id'].dtype)


numeric_cols = df.select_dtypes(include=['number']).columns

df[numeric_cols] = df.groupby('dataset')[numeric_cols]\
                    .transform(lambda x: x.fillna(x.median()))


string_cols = df.select_dtypes(include=['object', 'str']).columns
df[string_cols] = df[string_cols].fillna('Missing')



df = cap_outliers(df, numeric_cols)
df.to_csv(input_data, index=False)
## Just adding for the testing 
