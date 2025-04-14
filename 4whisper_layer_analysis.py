# use arena environment
# python -u 4whisper_layer_analysis.py --output_dir output > layer_analysis.log 2>&1
# python -u 4whisper_layer_analysis.py --output_dir output --figures > layer_analysis.log 2>&1
# python -u 4whisper_layer_analysis.py --output_dir output --metric g_val > layer_analysis.log 2>&1

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
from pathlib import Path

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

    parser.add_argument(
        "--metric",
        type=str,
        default="g_val",
        help="Metric to use for phoneme activations (default: g_val)"
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        help="Model variant to use (default: tiny)"
    )

    # example usage
    # python -u whisper_activations.py --phoneme_file phoneme_segments.pkl --output_dir output --block_index 2
    args = parser.parse_args()

    output_dir = args.output_dir
    do_figures = args.figures #or True
    DO_METRIC = args.metric
    VARIANT = args.variant

    print(type(do_figures), do_figures, flush=True)

    block_folders = glob.glob(f"{output_dir}/{VARIANT}/*")
    block_folders = [x for x in block_folders if os.path.isdir(x)]
    block_folders = [Path(x).as_posix() for x in block_folders]
    def get_index_from_path(path):
        # Extract the block index from the path
        return int(path.split('blocks[')[1].split(']')[0])
    
    blocks= [get_index_from_path(x) for x in block_folders]

    all_blocks_names = [
        f"{output_dir}/{VARIANT}/all_blocks_phoneme_vs_shuffled.pkl",
        f"{output_dir}/{VARIANT}/all_blocks_phoneme_vs_noise.pkl",
        f"{output_dir}/{VARIANT}/all_blocks_noise_vs_shuffled.pkl",
    ]

    block_data = {}
    block_data['phoneme2shuffled'] = []
    block_data['phoneme2noise'] = []
    block_data['noise2shuffled'] = []

    if not all([os.path.exists(x) for x in all_blocks_names]):
        for block_index in blocks:
            print(f"################# Processing block {block_index} ######################", flush=True)

            mlp_path = f"{output_dir}/{VARIANT}/model.encoder.blocks[{block_index}].mlp"
            df = pd.read_pickle(f"{mlp_path}/phoneme_activations.pkl")
            phoneme_vs_shuffled_file = f"{mlp_path}/phoneme_vs_shuffled.pkl"
            phoneme_vs_noise_file = f"{mlp_path}/phoneme_vs_noise.pkl"
            noise_vs_shuffled_file = f"{mlp_path}/noise_vs_shuffled.pkl"
            assert os.path.exists(phoneme_vs_shuffled_file), f"File {phoneme_vs_shuffled_file} does not exist"
            assert os.path.exists(phoneme_vs_noise_file), f"File {phoneme_vs_noise_file} does not exist"
            assert os.path.exists(noise_vs_shuffled_file), f"File {noise_vs_shuffled_file} does not exist"
            print(f"Loading existing dataframes for block {block_index}", flush=True)
            df_phoneme_vs_shuffled = pd.read_pickle(f"{mlp_path}/phoneme_vs_shuffled.pkl")
            df_phoneme_vs_noise = pd.read_pickle(f"{mlp_path}/phoneme_vs_noise.pkl")
            df_noise_vs_shuffled = pd.read_pickle(f"{mlp_path}/noise_vs_shuffled.pkl")
            print(f"Loaded phoneme vs shuffled and noise dataframes for block {block_index}", flush=True)

            df_phoneme_vs_shuffled['block_index'] = block_index
            df_phoneme_vs_noise['block_index'] = block_index
            df_noise_vs_shuffled['block_index'] = block_index
            block_data['phoneme2shuffled'].append(df_phoneme_vs_shuffled)
            block_data['phoneme2noise'].append(df_phoneme_vs_noise)
            block_data['noise2shuffled'].append(df_noise_vs_shuffled)

        df_phoneme2shuffled = pd.concat(block_data['phoneme2shuffled'])
        df_phoneme2noise = pd.concat(block_data['phoneme2noise'])
        df_noise2shuffled = pd.concat(block_data['noise2shuffled'])

        df_phoneme2noise.columns

        # save phoneme vs shuffled and noise dataframes
        df_phoneme2shuffled.to_pickle(f"{output_dir}/{VARIANT}/all_blocks_phoneme_vs_shuffled.pkl")
        df_phoneme2noise.to_pickle(f"{output_dir}/{VARIANT}/all_blocks_phoneme_vs_noise.pkl")
        df_noise2shuffled.to_pickle(f"{output_dir}/{VARIANT}/all_blocks_noise_vs_shuffled.pkl")
    else:
        print(f"Loading existing dataframes", flush=True)
        df_phoneme2shuffled = pd.read_pickle(all_blocks_names[0])
        df_phoneme2noise = pd.read_pickle(all_blocks_names[1])
        df_noise2shuffled = pd.read_pickle(all_blocks_names[2])
        print(f"Loaded phoneme vs shuffled and noise dataframes", flush=True)

    df_names = ['phoneme2shuffled', 'phoneme2noise', 'noise2shuffled']

    all_df_names = [f"{output_dir}/{VARIANT}/all_blocks_{df_}_{DO_METRIC}_phoneme.pkl" for df_ in df_names]

    if not all([os.path.exists(x) for x in all_df_names]):
        df_types = {}
        for df_this, df_name in zip([df_phoneme2shuffled, df_phoneme2noise, df_noise2shuffled], df_names):

            block_neuron_pairs = []

            for b in df_this.block_index.unique():
                df_this_b = df_this[df_this.block_index == b]
                print(f"Block {b} has {len(df_this_b)} phoneme activations", flush=True)

                for n in df_this_b['neuron'].unique():
                    df_this_b_n = df_this_b[df_this_b['neuron'] == n]
                    print(f"Neuron {n} has {len(df_this_b_n)} phoneme activations", flush=True)
                    block_neuron_pairs.append((b, n))
            
            block_neuron_dicts = []
            for b,n in block_neuron_pairs:
                df_this_b_n = df_this[(df_this.block_index == b) & (df_this.neuron == n)]
                print(f"Block {b} Neuron {n} has {len(df_this_b_n)} phoneme activations", flush=True)

                # get dvals sorted
                df_this_b_n_sorted=df_this_b_n.sort_values(DO_METRIC, ascending=False, inplace=False)
                #df_this_b_n_sorted[DO_METRIC]

                # Get the top positive phonemes before it reaches 60% of the max d_val

                if any(df_this_b_n_sorted[DO_METRIC] > 0):
                    max_d_val = df_this_b_n_sorted[DO_METRIC].max()
                    threshold = max_d_val * 0.7 #1-np.exp(-1) # 1-e^-1 = 0.6321
                    df_this_b_n_sorted_top = df_this_b_n_sorted[df_this_b_n_sorted[DO_METRIC] >= threshold]
                    num_positive = df_this_b_n_sorted[DO_METRIC][df_this_b_n_sorted[DO_METRIC] > 0].count()
                else:
                    df_this_b_n_sorted_top = df_this_b_n_sorted[df_this_b_n_sorted[DO_METRIC]  > 0] # this should be empty
                    num_positive = 0

                num_great_cohen = df_this_b_n_sorted[DO_METRIC][df_this_b_n_sorted[DO_METRIC] > 0.8].count()

                top_phonemes = df_this_b_n_sorted_top['phoneme'].unique()
                top_auc = df_this_b_n_sorted_top[DO_METRIC].sum()/(len(df_this_b_n_sorted_top['phoneme'].unique()) if len(df_this_b_n_sorted_top['phoneme'].unique()) > 0 else 1)
                # Get top negative phonemes before it reaches 60% of the min d_val (most control-confusing phonemes)
                top_df = df_this_b_n_sorted_top.copy()
                if any(df_this_b_n_sorted[DO_METRIC] < 0):
                    min_d_val = df_this_b_n_sorted[DO_METRIC].min()
                    threshold = min_d_val * 0.7 #1-np.exp(-1) # 1-e^-1 = 0.6321
                    df_this_b_n_sorted_bottom = df_this_b_n_sorted[df_this_b_n_sorted[DO_METRIC] <= threshold]
                    num_negative = df_this_b_n_sorted[DO_METRIC][df_this_b_n_sorted[DO_METRIC] < 0].count()
                else:
                    df_this_b_n_sorted_bottom = df_this_b_n_sorted[df_this_b_n_sorted[DO_METRIC] < 0] # this should be empty
                    num_negative
                
                num_worst_cohen = df_this_b_n_sorted[DO_METRIC][df_this_b_n_sorted[DO_METRIC] < -0.8].count()

                bottom_phonemes = df_this_b_n_sorted_bottom['phoneme'].unique()
                bottom_auc = df_this_b_n_sorted_bottom[DO_METRIC].sum()/(len(df_this_b_n_sorted_bottom['phoneme'].unique()) if len(df_this_b_n_sorted_bottom['phoneme'].unique()) > 0 else 1)
                bottom_df = df_this_b_n_sorted_bottom.copy()

                this_b_n_dict = {}
                this_b_n_dict['block_index'] = b
                this_b_n_dict['neuron'] = n
                this_b_n_dict['top_auc'] = top_auc
                this_b_n_dict['bottom_auc'] = bottom_auc
                this_b_n_dict['top_phonemes'] = top_phonemes
                this_b_n_dict['bottom_phonemes'] = bottom_phonemes
                this_b_n_dict['top_df'] = top_df
                this_b_n_dict['bottom_df'] = bottom_df
                this_b_n_dict['num_positive'] = num_positive
                this_b_n_dict['num_negative'] = num_negative
                this_b_n_dict['num_great_cohen'] = num_great_cohen
                this_b_n_dict['num_worst_cohen'] = num_worst_cohen
                this_b_n_dict['auc'] = df_this_b_n_sorted[DO_METRIC].sum()#/(len(df_this_b_n_sorted['phoneme'].unique()) if len(df_this_b_n_sorted['phoneme'].unique()) > 0 else 1)
                block_neuron_dicts.append(this_b_n_dict)

            df_block_neuron = pd.DataFrame(block_neuron_dicts)
            df_block_neuron['type'] = df_name
            df_block_neuron.to_pickle(f"{output_dir}/{VARIANT}/all_blocks_{df_name}_{DO_METRIC}_phoneme_activations.pkl")
            df_types[df_name] = df_block_neuron.copy()

        # number of top phonemes per neuron per block

        for df_ in df_types.keys():
            df_types[df_].to_pickle(f"{output_dir}/{VARIANT}/all_blocks_{df_}_{DO_METRIC}_phoneme.pkl")
    else:
        df_types = {}
        for df_ in df_names:
            df_types[df_] = pd.read_pickle(f"{output_dir}/{VARIANT}/all_blocks_{df_}_{DO_METRIC}_phoneme_activations.pkl")
            print(f"Loaded phoneme vs shuffled and noise dataframes for {df_}", flush=True)
    
    df_types[df_].iloc[0]
    df_types['phoneme2shuffled']['top_phonemes_count'] = df_types['phoneme2shuffled']['top_phonemes'].apply(lambda x: len(x))
    df_types['phoneme2noise']['top_phonemes_count'] = df_types['phoneme2noise']['top_phonemes'].apply(lambda x: len(x))
    df_types['noise2shuffled']['top_phonemes_count'] = df_types['noise2shuffled']['top_phonemes'].apply(lambda x: len(x))

    df_types['phoneme2shuffled']['bottom_phonemes_count'] = df_types['phoneme2shuffled']['bottom_phonemes'].apply(lambda x: len(x))
    df_types['phoneme2noise']['bottom_phonemes_count'] = df_types['phoneme2noise']['bottom_phonemes'].apply(lambda x: len(x))
    df_types['noise2shuffled']['bottom_phonemes_count'] = df_types['noise2shuffled']['bottom_phonemes'].apply(lambda x: len(x))

    if do_figures:
        groups = df_types['phoneme2shuffled'].groupby('block_index') # preview of the groups
        # plot number of top phonemes per neuron per block
        fig,axes = plt.subplots(len(df_types.keys()), len(groups), figsize=(20, 10),sharex=True, sharey=True)#,sharex=True, sharey=True)
        experiment_map = {'phoneme2shuffled':f'{DO_METRIC} of Phoneme vs Shuffled', 'phoneme2noise':f'{DO_METRIC} of Phoneme vs AM Noise', 'noise2shuffled':f'{DO_METRIC} of AM Noise vs Shuffled'}

        for i,dtype in enumerate(df_types.keys()):
            groups=df_types[dtype].groupby('block_index')

            for name, group in groups:
                j = int(name)
                sorted_top = group.sort_values('num_great_cohen', ascending=False)
                sorted_bottom = group.sort_values('bottom_phonemes_count', ascending=False)

                ax = axes[i][j]

                # Original line plot
                top_vals = sorted_top['num_great_cohen'].to_numpy()
                bottom_vals = sorted_top['num_worst_cohen'].to_numpy()

                ax.plot(bottom_vals, label=f'# Phonemes with {DO_METRIC} < -0.8')
                ax.plot(top_vals, label=f'# Phonemes with {DO_METRIC} > 0.8')

                # Mean lines
                ax.hlines(np.mean(bottom_vals), xmin=0, xmax=len(bottom_vals)-1, color='black', linestyle='--', label=f'Mean # Phonemes with {DO_METRIC} < -0.8')
                ax.hlines(np.mean(top_vals), xmin=0, xmax=len(top_vals)-1, color='red', linestyle='--', label=f'Mean # Phonemes with {DO_METRIC} > 0.8')

                # Slope (derivative)
                # top_deriv = np.diff(top_vals)
                # top_deriv = np.mean(top_deriv)  # mean slope
                # ax2 = ax.twinx()

                # ax2.hlines(top_deriv, xmin=0, xmax=len(top_vals)-1, color='blue', linestyle='--', label='Mean Slope')
                # ax2.set_ylabel("Δ top phoneme count")

                # # Optional: detect inflection points (drop > threshold)
                # threshold = -2  # tweak this
                # inflections = np.where(top_deriv < threshold)[0]
                # ax.scatter(inflections, top_vals[inflections], color='purple', label='Inflection Points', zorder=5)

                ax.set_xlabel('Neurons')
                ax.set_ylabel(f'Number of Phonemes per Neuron\n{experiment_map[dtype]}')
                ax.set_title(f'Block {name}')


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_{DO_METRIC}.png")
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_{DO_METRIC}.pdf")
        plt.close('all')

        # plot number of top phonemes per neuron per block
        fig,axes = plt.subplots(len(df_types.keys()), len(groups), figsize=(20, 10),sharex=True, sharey=True)#,sharex=True, sharey=True)
        experiment_map = {'phoneme2shuffled':f'{DO_METRIC} of Phoneme vs Shuffled', 'phoneme2noise':f'{DO_METRIC} of Phoneme vs AM Noise', 'noise2shuffled':f'{DO_METRIC} of AM Noise vs Shuffled'}

        for i,dtype in enumerate(df_types.keys()):
            groups=df_types[dtype].groupby('block_index')

            for name, group in groups:
                j = int(name)
                sorted_top = group.sort_values('num_great_cohen', ascending=False)
                sorted_bottom = group.sort_values('bottom_phonemes_count', ascending=False)

                ax = axes[i][j]

                # Original line plot
                top_vals = sorted_top['num_great_cohen'].to_numpy()
                bottom_vals = sorted_top['num_worst_cohen'].to_numpy()

                top_vals = top_vals / np.maximum(bottom_vals, np.ones_like(top_vals))
                top_vals=np.sort(top_vals, kind='mergesort')
                top_vals=top_vals[::-1]


                #ax.plot(bottom_vals, label=f'# Phonemes with {DO_METRIC} < -0.8')
                #ax.plot(top_vals, label=f'# Phonemes with {DO_METRIC} > 0.8')

                ax.plot(top_vals, label=f'# Phonemes with {DO_METRIC} > 0.8/# Phonemes with {DO_METRIC} < -0.8')

                # Mean lines
                #ax.hlines(np.mean(bottom_vals), xmin=0, xmax=len(bottom_vals)-1, color='black', linestyle='--', label=f'Mean # Phonemes with {DO_METRIC} < -0.8')
                ax.hlines(np.mean(top_vals), xmin=0, xmax=len(top_vals)-1, color='red', linestyle='--', label='Mean')

                # Slope (derivative)
                # top_deriv = np.diff(top_vals)
                # top_deriv = np.mean(top_deriv)  # mean slope
                # ax2 = ax.twinx()

                # ax2.hlines(top_deriv, xmin=0, xmax=len(top_vals)-1, color='blue', linestyle='--', label='Mean Slope')
                # ax2.set_ylabel("Δ top phoneme count")

                # # Optional: detect inflection points (drop > threshold)
                # threshold = -2  # tweak this
                # inflections = np.where(top_deriv < threshold)[0]
                # ax.scatter(inflections, top_vals[inflections], color='purple', label='Inflection Points', zorder=5)

                ax.set_xlabel('Neurons')
                ax.set_ylabel(f'Number of Phonemes per Neuron\n{experiment_map[dtype]}')
                ax.set_title(f'Block {name}')


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        #plt.show()
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_{DO_METRIC}_ratio.png")
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_{DO_METRIC}_ratio.pdf")

        # plot number of top phonemes per neuron per block
        fig,axes = plt.subplots(len(df_types.keys()), len(groups), figsize=(20, 10),sharex=True, sharey=True)#,sharex=True, sharey=True)
        experiment_map = {'phoneme2shuffled':f'{DO_METRIC} of Phoneme vs Shuffled', 'phoneme2noise':f'{DO_METRIC} of Phoneme vs AM Noise', 'noise2shuffled':f'{DO_METRIC} of AM Noise vs Shuffled'}
        for i,dtype in enumerate(df_types.keys()):
            groups=df_types[dtype].groupby('block_index')

            for name, group in groups:
                j = int(name)
                sorted_top = group.sort_values('auc', ascending=False)

                ax = axes[i][j]

                #ax.plot(bottom_vals, label=f'# Phonemes with {DO_METRIC} < -0.8')
                #ax.plot(top_vals, label=f'# Phonemes with {DO_METRIC} > 0.8')
                top_vals = sorted_top['auc'].to_numpy()
                ax.plot(top_vals, label=f'AUC of {DO_METRIC} across phonemes for each neuron')

                # Mean lines
                #ax.hlines(np.mean(bottom_vals), xmin=0, xmax=len(bottom_vals)-1, color='black', linestyle='--', label=f'Mean # Phonemes with {DO_METRIC} < -0.8')
                ax.hlines(np.mean(top_vals), xmin=0, xmax=len(top_vals)-1, color='red', linestyle='--', label='Mean')

                # Slope (derivative)
                # top_deriv = np.diff(top_vals)
                # top_deriv = np.mean(top_deriv)  # mean slope
                # ax2 = ax.twinx()

                # ax2.hlines(top_deriv, xmin=0, xmax=len(top_vals)-1, color='blue', linestyle='--', label='Mean Slope')
                # ax2.set_ylabel("Δ top phoneme count")

                # # Optional: detect inflection points (drop > threshold)
                # threshold = -2  # tweak this
                # inflections = np.where(top_deriv < threshold)[0]
                # ax.scatter(inflections, top_vals[inflections], color='purple', label='Inflection Points', zorder=5)

                ax.set_xlabel('Neurons')
                ax.set_ylabel(f'AUC across phonemes per Neuron\n{experiment_map[dtype]}')
                ax.set_title(f'Block {name}')


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_auc_{DO_METRIC}.png")
        fig.savefig(f"{output_dir}/{VARIANT}/phoneme_auc_{DO_METRIC}.pdf")

        plt.close('all')


if __name__ == "__main__":
    main()