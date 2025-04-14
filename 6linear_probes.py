import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from joblib import Parallel, delayed



VARIANT = "base"  # Change this to "base" or "large" as needed
output_dir = "output"
probe_dir = f"{output_dir}/{VARIANT}/phoneme_probes.pkl"

if not os.path.isfile(probe_dir):
    print(f"Probe file not found: {probe_dir}. Proceeding with computation.")

    block_folders = glob.glob(f"{output_dir}/{VARIANT}/*")
    block_folders = [x for x in block_folders if os.path.isdir(x)]
    block_folders = [Path(x).as_posix() for x in block_folders]
    def get_index_from_path(path):
        # Extract the block index from the path
        return int(path.split('blocks[')[1].split(']')[0])

    blocks= [get_index_from_path(x) for x in block_folders]
    block_activations = []
    for block in blocks:
        # Get the path to the block folder
        block_folder = block_folders[block]
        # Get the path to the activations file
        activations_file = f"{block_folder}/phoneme_activations.pkl"
        assert os.path.isfile(activations_file), f"File not found: {activations_file}"
        # Load the DataFrame
        df = pd.read_pickle(activations_file)
        mean_col = df.columns[['mlp_mean' in x for x in df.columns].index(True)]
        df2 = df[['phoneme',mean_col,'key','utterance']].copy()
        df2['block'] = block
        df2['activations']= df2[mean_col]
        df2['variant']= VARIANT
        df2.drop(columns=[mean_col], inplace=True)
        block_activations.append(df2)

    # Concatenate all DataFrames into one
    df = pd.concat(block_activations, ignore_index=True)



    df_orig     = df[~df['phoneme'].str.contains('@')]
    df_shuffled = df[df['phoneme'].str.contains('@shuffled')]
    df_noise    = df[df['phoneme'].str.contains('@noise')]

    df.iloc[0]

    def run_probe(label, df_variant, b):
        print(f"Processing block {b} for {label}...")

        dfb = df_variant[df_variant['block'] == b]
        if len(dfb) < 2:
            return None  # skip if not enough data

        try:
            X = np.stack(dfb['activations'].values)
            y_raw = dfb['phoneme'].str.replace(r'@.*', '', regex=True)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            scores = cross_val_score(clf, X, y, cv=5)

            return {
                'block': b,
                'variant': label,
                'accuracy': scores,
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'clf': clf,
                'model': df_variant['variant'].values[0],  # assuming column is 'variant'
            }
        except Exception as e:
            print(f"Error processing block {b} for {label}: {e}")
            return None

    # Variants to test
    variants = [('Original', df_orig), ('Shuffled', df_shuffled), ('AM Noise', df_noise)]

    # Launch in parallel (use n_jobs=-1 for all cores)
    njobs=8
    if njobs==1:
        probe_dicts = []
        for label, df_variant in variants:
            for b in sorted(df_variant['block'].unique()):
                print(f"Processing block {b} for {label}...")
                probe_dict = run_probe(label, df_variant, b)
                if probe_dict is not None:
                    probe_dicts.append(probe_dict)
    else:
        probe_dicts = Parallel(n_jobs=8)(
            delayed(run_probe)(label, df_variant, b)
            for label, df_variant in variants
            for b in sorted(df_variant['block'].unique())
        )

    # Filter out None results
    assert None not in probe_dicts, "Some probes returned None. Check the logs."

    # Convert to DataFrame
    df_probes = pd.DataFrame(probe_dicts)

    # Save the DataFrame to a pickle file

    df_probes.to_pickle()
    df_probes.to_csv(f"{output_dir}/{VARIANT}/phoneme_probes.csv", index=False)

df_probes = pd.read_pickle(f"{output_dir}/{VARIANT}/phoneme_probes.pkl")

import matplotlib.pyplot as plt
import seaborn as sns

# Clean up the dataframe
df_probes['block'] = df_probes['block'].apply(lambda x: str(x).replace('blocks[', '').replace(']', ''))

# Sort blocks numerically
df_probes['block'] = pd.Categorical(df_probes['block'], ordered=True, categories=sorted(df_probes['block'].unique()))

# Ensure variant order
df_probes['variant'] = pd.Categorical(df_probes['variant'], categories=['Original', 'Shuffled', 'AM Noise'], ordered=True)

# Plot
plt.figure(figsize=(8, 5))
for variant in df_probes['variant'].unique():
    df_v = df_probes[df_probes['variant'] == variant]
    blocks = df_v['block'].values
    means = df_v['mean_accuracy'].values
    stds = df_v['std_accuracy'].values

    plt.errorbar(
        blocks,
        means,
        yerr=stds,
        label=variant,
        marker='o',
        capsize=5,
        linestyle='-'
    )

plt.ylim(0, 1)
plt.title(f"Linear Probe Accuracy per Block (Whisper {VARIANT})")
plt.xlabel("Encoder Block")
plt.ylabel("Mean Accuracy ± Std. Dev.")
plt.grid(True)
plt.legend(title="Variant")
plt.tight_layout()

plt.savefig(f"{output_dir}/{VARIANT}/phoneme_probes.png", dpi=300)
plt.savefig(f"{output_dir}/{VARIANT}/phoneme_probes.pdf", dpi=300)
#plt.show()
plt.close('all')
