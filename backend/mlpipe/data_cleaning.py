import pandas as pd
import os
from pathlib import Path

input_data = "./datasets/inputFile.csv"

df = pd.read_csv(input_data, low_memory=False)

print(len(df))

