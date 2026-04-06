import pandas as pd
import numpy as np
import os
from pathlib import Path

datapath = './datasets/Indian-Healthcare-Symptom-Disease-Dataset - Sheet1 (2).csv'
all_dataframe = []

# columns to Severity,Average Duration (days),Common in Region,Language Availability
columns_to_drop = ['Average Duration (days)',' Common in Region', 'Language Availability']

Important_columns = ['Symptom', 'Possible Diseases','Severity']

# This should be apply first 



df = pd.read_csv(datapath, low_memory= False)
cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

        
'''
    {
       Malria: {Fever, JP, Insomina}
       TB: {fati}
       Disease : {Symptoms}

       row: dis    Colum Symtom

       M:    F JP IN
             1 1  1 0 0 0
       TB    1  1  0  0 0    
    }
'''        
        


def data_generalization(df):
    df['Possible Diseases'] = df['Possible Diseases'].apply(
        lambda x: [d.strip().lower() for d in x.split(',')]
    )

    df['Symptom'] = df['Symptom'].str.strip().str.lower()
    df['Severity'] = df['Severity'].str.strip().str.lower()

    # severity mapping
    severity_map = {
        'mild': 1,
        'moderate': 2,
        'severe': 3
    }

    df['Severity_Num'] = df['Severity'].map(severity_map).fillna(1)

    # normalize duration
    df['Duration_Norm'] = df['Average Duration (days)'] / df['Average Duration (days)'].max()

    # combine weight
    df['Weight'] = df['Severity_Num'] * (1 + df['Duration_Norm'])

    # explode
    df_exploded = df.explode('Possible Diseases')
    df_exploded = df_exploded.rename(columns={'Possible Diseases': 'Disease'})

    # weighted matrix
    matrix = pd.pivot_table(
        df_exploded,
        index='Disease',
        columns='Symptom',
        values='Weight',
        aggfunc='max',
        fill_value=0
    )
    
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    matrix.to_csv('weighted_matrix.csv')
    


# def save_the_end_result(df)
data_generalization(pd.read_csv(datapath,low_memory=False))

