# use arena environment
# python -u whisper_activation_analysis.py --output_dir output > activation_analysis.log 2>&1
# python -u whisper_activation_analysis.py --output_dir output --figures > activation_analysis.log 2>&1


import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

import joblib
from joblib import Parallel, delayed

import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt 
import argparse
import itertools
import glob


def compare_and_format(args):
    phoneme_real, phoneme_control, control_type, df, agg, activations_key_frames_x_neurons, njobs = args
    print(f"Comparing {phoneme_real} vs {phoneme_control}", flush=True)
    result = extract_phoneme_vs_control(df, phoneme_real, phoneme_control, agg=agg, activations_key_frames_x_neurons=activations_key_frames_x_neurons,njobs=njobs)
    result['phoneme'] = phoneme_real
    result['phoneme_control'] = phoneme_control
    result['control_type'] = control_type
    return pd.DataFrame.from_dict(result)

from tqdm.contrib.concurrent import process_map

def build_control_comparisons(df, real_phonemes, shuffled_phonemes, noise_phonemes, agg=np.mean, activations_key_frames_x_neurons=None, outer_jobs=1,inner_jobs=1):
    tasks = (
        [(r, s, 'shuffled', df, agg, activations_key_frames_x_neurons,inner_jobs) for r, s in zip(real_phonemes, shuffled_phonemes)] +
        [(r, n, 'noise', df, agg, activations_key_frames_x_neurons,inner_jobs) for r, n in zip(real_phonemes, noise_phonemes)]
    )

    if outer_jobs == 1:
        all_dfs = [compare_and_format(task) for task in tasks]
    else:
        all_dfs = process_map(compare_and_format, tasks, max_workers=outer_jobs, desc="Comparing phonemes")

    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

def get_neuron_vs_phoneme_matrix(df,stat_key):
    df_ = df.copy()
    # Convert the list-of-arrays column into a matrix
    df_['activ_array'] = df_[stat_key].apply(lambda x: np.array(x) if x is not None else np.zeros(384))

    # Group by phoneme
    grouped = df_.groupby('phoneme')['activ_array']

    # Compute mean distribution value per neuron for each phoneme
    phoneme_neuron_agg = grouped.apply(lambda x: np.vstack(x).mean(axis=0))  # shape: (num_phonemes, 384)


    # Convert to DataFrame
    neuron_df = pd.DataFrame(phoneme_neuron_agg.tolist(), 
                            index=phoneme_neuron_agg.index)

    # For each neuron (column), find the phoneme with the highest mean activation
    top_phoneme_per_neuron = neuron_df.idxmax(axis=0)  # Series of length 384
    top_activation_value = neuron_df.max(axis=0)

    def selectivity_score(vec):
        sorted_vals = np.sort(vec)[::-1]
        return (sorted_vals[0] - sorted_vals[1]) / (sorted_vals[0] + 1e-6)  # avoid div-by-zero

    selectivity = neuron_df.apply(selectivity_score, axis=0)

    output_dict = {}
    output_dict['phoneme_neuron_agg'] = phoneme_neuron_agg
    output_dict['neuron_df'] = neuron_df
    output_dict['top_phoneme_per_neuron'] = top_phoneme_per_neuron
    output_dict['top_activation_value'] = top_activation_value
    output_dict['selectivity'] = selectivity


    return output_dict

def get_phoneme_vs_phoneme_per_neuron_matrix(df, stat_key, neuron_idx):

    df_ = df.copy()
    phoneme_groups = defaultdict(list)

    for _, row in df_.iterrows():
        vec = row[stat_key]
        if vec is not None:
            phoneme_groups[row['phoneme']].append(vec[neuron_idx])

    # Convert to arrays
    phoneme_activs = {k: np.array(v) for k, v in phoneme_groups.items() if len(v) >= 1} # at least 1 sample


    phonemes = sorted(phoneme_activs.keys())
    matrix = pd.DataFrame(index=phonemes, columns=phonemes)

    matrix_t = pd.DataFrame(index=phonemes, columns=phonemes)
    matrix_p = pd.DataFrame(index=phonemes, columns=phonemes)
    matrix_d = pd.DataFrame(index=phonemes, columns=phonemes)
    def cohens_d(x, y):
        return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

    for i, p1 in enumerate(phonemes):
        for j, p2 in enumerate(phonemes):
            if i >= j:
                continue  # avoid redundant comparisons
            a1 = phoneme_activs[p1]
            a2 = phoneme_activs[p2]

            t, p = ttest_ind(a1, a2, equal_var=False)
            d = cohens_d(phoneme_activs[p1], phoneme_activs[p2])
            matrix_t.at[p1, p2] = t
            matrix_t.at[p2, p1] = t
            matrix_p.at[p1, p2] = p
            matrix_p.at[p2, p1] = p
            matrix_d.at[p1, p2] = d
            matrix_d.at[p2, p1] = -d
            matrix.at[p1, p2] = (t,p,d)
            matrix.at[p2, p1] = (t,p,-d)
    
    output_dict = {}
    output_dict['matrix'] = matrix
    output_dict['matrix_t'] = matrix_t
    output_dict['matrix_p'] = matrix_p
    output_dict['matrix_d'] = matrix_d
    print(neuron_idx, flush=True)
    return output_dict


def get_phoneme_vs_phoneme_all_neurons(df, stat_key,njobs=1):
    phoneme_vs_phoneme_matrix = {}
    num_neurons = df[stat_key].iloc[0].shape[0]

    if njobs == 1:
        for neuron_idx in range(num_neurons):
            print(f"Processing neuron {neuron_idx}", flush=True)
            phoneme_vs_phoneme_matrix[neuron_idx] = get_phoneme_vs_phoneme_per_neuron_matrix(df, stat_key, neuron_idx)
    else:
        phoneme_vs_phoneme_matrix_ = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(get_phoneme_vs_phoneme_per_neuron_matrix)(df, stat_key, neuron_idx)
            for neuron_idx in range(num_neurons)
        )
        for neuron_idx in range(num_neurons):
            phoneme_vs_phoneme_matrix[neuron_idx] = phoneme_vs_phoneme_matrix_[neuron_idx]
    return phoneme_vs_phoneme_matrix

def cohens_d(x, y):
    """Calculate Cohen's d between two arrays."""
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2 + 1e-6)

def cohens_d_paired(a, b):
    diffs = np.array(a) - np.array(b)
    return np.mean(diffs) / np.std(diffs, ddof=1)

def hedges_g_paired(a, b):
    d = cohens_d_paired(a, b)
    n = len(a)
    correction = 1 - (3 / (4 * n - 1))
    return d * correction

def compute_stats_for_neuron(i, mat_a, mat_b):
    a_col = mat_a[:, i]
    b_col = mat_b[:, i]

    # Statistical tests
    t, p = ttest_rel(a_col, b_col, alternative="greater")
    d = cohens_d_paired(a_col, b_col)
    hedges_g = hedges_g_paired(a_col, b_col)

    # Sample-level sign comparison
    sign_a = np.sign(a_col)
    sign_b = np.sign(b_col)

    tp = np.sum((sign_a == 1) & (sign_b == 1))
    tn = np.sum((sign_a == -1) & (sign_b == -1))
    fp = np.sum((sign_a == -1) & (sign_b == 1))
    fn = np.sum((sign_a == 1) & (sign_b == -1))
    zero_mismatch = np.sum((sign_a == 0) != (sign_b == 0))

    total = len(sign_a)
    match = np.sum(sign_a == sign_b)
    mismatch = np.sum(sign_a != sign_b)

    return {
        'neuron': i,
        't_val': t,
        'p_val': p,
        'd_val': d,
        'g_val': hedges_g,
        'sign_a': np.sign(np.mean(a_col)),
        'sign_b': np.sign(np.mean(b_col)),
        'median_a': np.median(a_col),
        'median_b': np.median(b_col),
        'mean_a': np.mean(a_col),
        'mean_b': np.mean(b_col),
        'std_a': np.std(a_col),
        'std_b': np.std(b_col),
        'max_a': np.max(a_col),
        'max_b': np.max(b_col),
        'min_a': np.min(a_col),
        'min_b': np.min(b_col),
        'q25_a': np.quantile(a_col, 0.25),
        'q25_b': np.quantile(b_col, 0.25),
        'q75_a': np.quantile(a_col, 0.75),
        'q75_b': np.quantile(b_col, 0.75),
        'q90_a': np.quantile(a_col, 0.90),
        'q90_b': np.quantile(b_col, 0.90),
        'q10_a': np.quantile(a_col, 0.10),
        'q10_b': np.quantile(b_col, 0.10),

        # Sign mismatch stats
        'sign_match_count': match,
        'sign_mismatch_count': mismatch,
        'sign_match_rate': match / total,
        'sign_mismatch_rate': mismatch / total,
        'sign_a+b+': tp,
        'sign_a-b-': tn,
        'sign_a-b+': fp,
        'sign_a+b-': fn,
        'sign_zero_mismatch': zero_mismatch
    }

def extract_phoneme_vs_control(df, phoneme_a, phoneme_b, agg=np.mean, activations_key_frames_x_neurons='activations_model.encoder.blocks[2].mlp_list',njobs=1):

    #df['phoneme']
    phonemes_a = [phoneme_a]
    phonemes_b = [phoneme_b]
    df_a = df[df['phoneme'].isin(phonemes_a)].copy()
    df_b = df[df['phoneme'].isin(phonemes_b)].copy()

    # sort by key
    df_a = df_a.sort_values(by='utterance')
    df_b = df_b.sort_values(by='utterance')

    # Check if the keys are the same
    assert all(df_a['utterance'].to_numpy()==df_b['utterance'].to_numpy())

    # Apply frame aggregation
    df_a['neuron_vec'] = df_a[activations_key_frames_x_neurons].apply(lambda x: agg(x, axis=0) if isinstance(x, np.ndarray) else None)
    df_b['neuron_vec'] = df_b[activations_key_frames_x_neurons].apply(lambda x: agg(x, axis=0) if isinstance(x, np.ndarray) else None)

    # Drop rows with None values in neuron_vec
    df_a = df_a.dropna(subset=['neuron_vec'])
    df_b = df_b.dropna(subset=['neuron_vec'])

    assert all(df_a['utterance'].to_numpy()==df_b['utterance'].to_numpy())

    # Stack into matrices (samples of that phoneme × neurons)
    mat_a = np.stack(df_a['neuron_vec'].values)
    mat_b = np.stack(df_b['neuron_vec'].values)


    n_neurons = mat_a.shape[1]
    results = Parallel(n_jobs=njobs, verbose=0)(
        delayed(compute_stats_for_neuron)(i, mat_a, mat_b) for i in range(n_neurons)
    )
    df_neuron_stats = pd.DataFrame(results)
    return df_neuron_stats


def plot_sorted_phoneme_metric(
    neuron_df: pd.DataFrame,
    metric: str = "d_vals",  # or "t_vals", "p_vals"
    kind: str = "bar",       # "bar" or "box"
    title: str = None,
    save_path: str = None,
    plot: bool = False,
    neg_log: bool = False,
):
    df = neuron_df.copy()
    
    # Strip @noise or @shuffled if needed
    df["base_phoneme"] = df["phoneme"].apply(lambda p: p.split("@")[0])

    if neg_log:
        df[metric] = -np.log10(df[metric])

    # Aggregate by base phoneme (e.g., mean across comparisons)
    agg = df.groupby("base_phoneme")[metric].mean().sort_values(ascending=False)

    # Reorder dataframe
    df["base_phoneme"] = pd.Categorical(df["base_phoneme"], categories=agg.index, ordered=True)

    # Plot
    plt.figure(figsize=(18, 6))
    if kind == "bar":
        sns.barplot(x="base_phoneme", y=metric, data=df, estimator="mean", errorbar=None, hue='base_phoneme', palette="Spectral")
    elif kind == "box":
        sns.boxplot(x="base_phoneme", y=metric, data=df, palette="Spectral")

    plt.xticks(rotation=90)
    plt.xlabel("Phoneme")
    plt.ylabel(metric.replace("_", " ").capitalize() + (" (-log10)" if neg_log else ""))
    plt.title(title or f"Sorted Phoneme {metric.replace('_', ' ').capitalize()}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if plot:
        plt.show()
    plt.close()

def main():

    parser = argparse.ArgumentParser(description="Extract and analyze phoneme activations from Whisper.")

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output", 
        help="Directory where activations outputs were saved (default: output)"
    )

    parser.add_argument(
        "--figures",
        action="store_true",
        dest="figures",
        default=False,
        help="Generate figures for phoneme activations (default: False)"
    )

    # example usage
    # python -u whisper_activations.py --phoneme_file phoneme_segments.pkl --output_dir output --block_index 2
    args = parser.parse_args()

    output_dir = args.output_dir
    do_figures = args.figures

    print(type(do_figures), do_figures, flush=True)

    block_folders = glob.glob(f"{output_dir}/*")
    block_folders = [x for x in block_folders if os.path.isdir(x)]
    def get_index_from_path(path):
        # Extract the block index from the path
        return int(path.split('blocks[')[1].split(']')[0])
    
    blocks= [get_index_from_path(x) for x in block_folders]

    for block_index in blocks:
        print(f"################# Processing block {block_index} ######################", flush=True)

        mlp_path = f"{output_dir}/model.encoder.blocks[{block_index}].mlp"
        df = pd.read_pickle(f"{mlp_path}/phoneme_activations.pkl")


        if not os.path.isfile(f"{mlp_path}/phoneme_vs_shuffled.pkl") or not os.path.isfile(f"{mlp_path}/phoneme_vs_noise.pkl"):

            min_samples = 10

            valid_phonemes = df['phoneme'].value_counts()
            valid_phonemes = valid_phonemes[valid_phonemes >= min_samples].index.tolist()
            invalid_phonemes = df['phoneme'].value_counts()
            invalid_phonemes = invalid_phonemes[invalid_phonemes < min_samples].index.tolist()
            df['phoneme'].value_counts().to_csv(f"{mlp_path}/phoneme_counts.csv")

            df = df[df['phoneme'].isin(valid_phonemes)]

            print(f"Valid phonemes: {valid_phonemes}", flush=True)
            print(f"Invalid phonemes: {invalid_phonemes}", flush=True)

            # save value counts to file
            #df[df['phoneme'].isin(['eng'])].copy().iloc[0]['activations_model.encoder.blocks[2].mlp_list']

            # low size dataframe
            #df = df.drop(columns=[x for x in df.columns if any([y in x for y in ['segment','max','min','std','median','quantile','mean']])])

            # save the dataframe to a new file
            #df.to_pickle(f"{mlp_path}/phoneme_activations_clean.pkl")

            #df.iloc[0]

            #df.iloc[0]['activations_model.encoder.blocks[2].mlp_list'].shape
            activations_key_frames_x_neurons = f'activations_model.encoder.blocks[{block_index}].mlp_list'

            # real phonemes
            real_phonemes = df['phoneme'].unique()
            real_phonemes = [x for x in real_phonemes if not any([y in x for y in ['noise','shuffled']])]

            shuffled_phonemes = df['phoneme'].unique()
            shuffled_phonemes = [x for x in shuffled_phonemes if any([y in x for y in ['shuffled']])]

            noise_phonemes = df['phoneme'].unique()
            noise_phonemes = [x for x in noise_phonemes if any([y in x for y in ['noise']])]

            # sort the phonemes
            real_phonemes = sorted(real_phonemes)
            shuffled_phonemes = [f"{p}@shuffled" for p in real_phonemes]
            noise_phonemes = [f"{p}@noise" for p in real_phonemes]

            # assert same order
            assert [x+'@shuffled' == y for x,y in zip(real_phonemes,shuffled_phonemes)]
            assert [x+'@noise' for x in real_phonemes] == noise_phonemes

            df_phoneme_comparisons = build_control_comparisons(
                df=df,
                real_phonemes=real_phonemes,
                shuffled_phonemes=shuffled_phonemes,
                noise_phonemes=noise_phonemes,
                agg=np.mean,
                activations_key_frames_x_neurons=activations_key_frames_x_neurons,
                outer_jobs=1,
                inner_jobs=8,
            )

            # Save
            df_phoneme_comparisons.to_pickle(f"{mlp_path}/phoneme_vs_controls.pkl")
            df_phoneme_vs_shuffled = df_phoneme_comparisons.query("control_type == 'shuffled'")
            df_phoneme_vs_noise = df_phoneme_comparisons.query("control_type == 'noise'")
            df_phoneme_vs_shuffled.to_pickle(f"{mlp_path}/phoneme_vs_shuffled.pkl")
            df_phoneme_vs_noise.to_pickle(f"{mlp_path}/phoneme_vs_noise.pkl")
        else:
            print(f"Loading existing phoneme vs shuffled and noise dataframes for block {block_index}", flush=True)
            df_phoneme_vs_shuffled = pd.read_pickle(f"{mlp_path}/phoneme_vs_shuffled.pkl")
            df_phoneme_vs_noise = pd.read_pickle(f"{mlp_path}/phoneme_vs_noise.pkl")
            df_phoneme_comparisons = pd.read_pickle(f"{mlp_path}/phoneme_vs_controls.pkl")

        if do_figures:
            print(f"Generating figures for block {block_index}", flush=True)
            # Create output folder if it doesn't exist
            os.makedirs(f"{mlp_path}/figures", exist_ok=True)

            # Define the task for a single neuron/metric pair
            def plot_neuron_metric(neuron_idx, metric,df_phoneme_vs_shuffled, df_phoneme_vs_noise,block_index,experiment='phoneme2shuffled'):
                if experiment == 'phoneme2shuffled':
                    df = df_phoneme_vs_shuffled.copy()
                else:
                    df = df_phoneme_vs_noise.copy()
                neuron_df = df[df['neuron'] == neuron_idx].copy()

                # ignore phonemes with nan none values
                nas = neuron_df[metric].isna()
                nones = neuron_df[metric] == None
                infs = neuron_df[metric] == np.inf
                neg_infs = neuron_df[metric] == -np.inf
                dropped_df = neuron_df[(nas | nones | infs | neg_infs)]
                neuron_df = neuron_df[~(nas | nones | infs | neg_infs)]
                print(f"Processing neuron {neuron_idx}, metric {metric}", flush=True)

                print('Dropped:', flush=True)
                for idx, row in dropped_df.iterrows():
                    print(f"{row['phoneme']} {row['phoneme_control']} {row[metric]}", flush=True)

                neg_log = metric == 'p_vals'
                plot_sorted_phoneme_metric(
                    neuron_df,
                    metric=metric,
                    kind="bar",
                    title=f"Sorted Phoneme {metric.replace('_',' ')} between {experiment.replace('2',' vs ')} (Neuron {neuron_idx})",
                    save_path=f"{mlp_path}/figures/{metric}/block-{block_index}_metric-{metric}_neuron-{neuron_idx}_exp-{experiment}.png",
                    plot=False,
                    neg_log=neg_log,
                )

            # Generate all combinations
            experiments = ['phoneme2shuffled','phoneme2noise']
            neuron_ids = df_phoneme_vs_shuffled['neuron'].unique()
            metrics = ['d_vals'] # 't_vals', 'p_vals'

            for metric in metrics:
                os.makedirs(f"{mlp_path}/figures/{metric}", exist_ok=True)

            combinations = list(itertools.product(neuron_ids, metrics, experiments))

            # Run in parallel
            Parallel(n_jobs=8)(
                delayed(plot_neuron_metric)(neuron_idx, metric, df_phoneme_vs_shuffled, df_phoneme_vs_noise, block_index, experiment=exp)
                for neuron_idx, metric,exp in combinations
            )

if __name__ == "__main__":
    main()