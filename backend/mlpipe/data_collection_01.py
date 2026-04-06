import pandas as pd
import numpy as np
import os
from pathlib import Path

datapath = './datasets/Indian-Healthcare-Symptom-Disease-Dataset - Sheet1 (2).csv'
all_dataframe = []

# columns to Severity,Average Duration (days),Common in Region,Language Availability
columns_to_drop = ['Severity', 'Average Duration (days)',' Common in Region', 'Language Availability']

Important_columns = ['Symptom', 'Possible Diseases']

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
    df['Possible Diseases'] = df['Possible Diseases'].apply(lambda x: [d.strip().lower() for d in x.split(',')])
    df['Symptom'] = df['Symptom'].str.strip().str.lower()

    df_exploded = df.explode('Possible Diseases')
    df_exploded = df_exploded.rename(columns={'Possible Diseases': 'Disease'}).reset_index(drop=True)

    print(df_exploded['Disease'])
    print(df_exploded['Symptom'])
    matrix = pd.crosstab(df_exploded['Symptom'],df_exploded['Disease'])
    
    matrix.to_csv('symptom_diseases_Matrix.csv')
    


# def save_the_end_result(df)
data_generalization(pd.read_csv(datapath,low_memory=False))

