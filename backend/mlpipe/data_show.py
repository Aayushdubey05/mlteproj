import pandas as pd 
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns


df  =pd.read_csv("weighted_matrix.csv")

dfn = df.drop(columns=['Disease'])

mp.figure(figsize=(20,10))
sns.heatmap(dfn, cmap="viridis")
mp.title("Disease vs Symtoms")
mp.show()
