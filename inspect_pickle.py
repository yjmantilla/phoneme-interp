import numpy as np
activations = np.load(r'output\model.encoder.blocks[2].mlp\segment_activations.npy',allow_pickle=True)

activations.shape

import pandas as pd

df=pd.read_pickle(r'output\tiny\model.encoder.blocks[2].mlp\phoneme_activations.pkl')

df.columns

df.iloc[0]

# go to neuron 1

neuron_index = 1
df1 = df.copy()

array_cols = [x for x in df1.columns if 'activations' in x]

for col in array_cols:
    df1[col] = df1[col].apply(lambda x: x[neuron_index] if isinstance(x, np.ndarray) and len(x) > neuron_index else None)

# drop noise and shuffled phonemes

rows_to_drop = df1[df1['phoneme'].str.contains('noise', na=False) | df1['phoneme'].str.contains('shuffled', na=False)].index

df1.drop(rows_to_drop, inplace=True)
df1['segment'] = None
# do a boxplot of the activations for each phoneme in the same figure with rainbow colors, sorted by mean value
import matplotlib.pyplot as plt
import seaborn as sns

for i, array_col in enumerate(array_cols):
    # Calculate mean activation for each phoneme
    mean_values = df1.groupby('phoneme')[array_col].mean().sort_values(ascending=False)
    sorted_phonemes = mean_values.index

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df1, x='phoneme', y=array_col, palette='rainbow', order=sorted_phonemes)
    plt.title(f'Boxplot of {array_col} for each phoneme (sorted by mean value in descending order)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()