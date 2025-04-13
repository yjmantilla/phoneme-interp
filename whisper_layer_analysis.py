# use arena environment
# python -u whisper_layer_analysis.py --output_dir output > layer_analysis.log 2>&1
# python -u whisper_layer_analysis.py --output_dir output --figures > layer_analysis.log 2>&1


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


    block_data = {}
    block_data['phoneme2shuffled'] = []
    block_data['phoneme2noise'] = []
    for block_index in blocks:
        print(f"################# Processing block {block_index} ######################", flush=True)

        mlp_path = f"{output_dir}/model.encoder.blocks[{block_index}].mlp"
        df = pd.read_pickle(f"{mlp_path}/phoneme_activations.pkl")
        phoneme_vs_shuffled_file = f"{mlp_path}/phoneme_vs_shuffled.pkl"
        phoneme_vs_noise_file = f"{mlp_path}/phoneme_vs_noise.pkl"
        assert os.path.exists(phoneme_vs_shuffled_file), f"File {phoneme_vs_shuffled_file} does not exist"
        assert os.path.exists(phoneme_vs_noise_file), f"File {phoneme_vs_noise_file} does not exist"
        print(f"Loading existing phoneme vs shuffled and noise dataframes for block {block_index}", flush=True)
        df_phoneme_vs_shuffled = pd.read_pickle(f"{mlp_path}/phoneme_vs_shuffled.pkl")
        df_phoneme_vs_noise = pd.read_pickle(f"{mlp_path}/phoneme_vs_noise.pkl")
        print(f"Loaded phoneme vs shuffled and noise dataframes for block {block_index}", flush=True)

        df_phoneme_vs_shuffled['block_index'] = block_index
        df_phoneme_vs_noise['block_index'] = block_index
        block_data['phoneme2shuffled'].append(df_phoneme_vs_shuffled)
        block_data['phoneme2noise'].append(df_phoneme_vs_noise)

    df_phoneme2shuffled = pd.concat(block_data['phoneme2shuffled'])
    df_phoneme2noise = pd.concat(block_data['phoneme2noise'])

    df_phoneme2noise.columns

    # save phoneme vs shuffled and noise dataframes
    df_phoneme2shuffled.to_pickle(f"{output_dir}/all_blocks_phoneme_vs_shuffled.pkl")
    df_phoneme2noise.to_pickle(f"{output_dir}/all_blocks_phoneme_vs_noise.pkl")

    df_types = {}
    for df_this, df_name in zip([df_phoneme2shuffled, df_phoneme2noise], ['phoneme2shuffled', 'phoneme2noise']):

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
            df_this_b_n_sorted=df_this_b_n.sort_values('d_val', ascending=False, inplace=False)
            #df_this_b_n_sorted['d_val']

            # Get the top positive phonemes before it reaches 60% of the max d_val

            if any(df_this_b_n_sorted['d_val'] > 0):
                max_d_val = df_this_b_n_sorted['d_val'].max()
                threshold = max_d_val * 0.7 #1-np.exp(-1) # 1-e^-1 = 0.6321
                df_this_b_n_sorted_top = df_this_b_n_sorted[df_this_b_n_sorted['d_val'] >= threshold]
                num_positive = df_this_b_n_sorted['d_val'][df_this_b_n_sorted['d_val'] > 0].count()
            else:
                df_this_b_n_sorted_top = df_this_b_n_sorted[df_this_b_n_sorted['d_val']  > 0] # this should be empty
                num_positive = 0

            num_great_cohen = df_this_b_n_sorted['d_val'][df_this_b_n_sorted['d_val'] > 0.8].count()

            top_phonemes = df_this_b_n_sorted_top['phoneme'].unique()
            top_auc = df_this_b_n_sorted_top['d_val'].sum()/(len(df_this_b_n_sorted_top['phoneme'].unique()) if len(df_this_b_n_sorted_top['phoneme'].unique()) > 0 else 1)
            # Get top negative phonemes before it reaches 60% of the min d_val (most control-confusing phonemes)
            top_df = df_this_b_n_sorted_top.copy()
            if any(df_this_b_n_sorted['d_val'] < 0):
                min_d_val = df_this_b_n_sorted['d_val'].min()
                threshold = min_d_val * 0.7 #1-np.exp(-1) # 1-e^-1 = 0.6321
                df_this_b_n_sorted_bottom = df_this_b_n_sorted[df_this_b_n_sorted['d_val'] <= threshold]
                num_negative = df_this_b_n_sorted['d_val'][df_this_b_n_sorted['d_val'] < 0].count()
            else:
                df_this_b_n_sorted_bottom = df_this_b_n_sorted[df_this_b_n_sorted['d_val'] < 0] # this should be empty
                num_negative
            
            num_worst_cohen = df_this_b_n_sorted['d_val'][df_this_b_n_sorted['d_val'] < -0.8].count()

            bottom_phonemes = df_this_b_n_sorted_bottom['phoneme'].unique()
            bottom_auc = df_this_b_n_sorted_bottom['d_val'].sum()/(len(df_this_b_n_sorted_bottom['phoneme'].unique()) if len(df_this_b_n_sorted_bottom['phoneme'].unique()) > 0 else 1)
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
            block_neuron_dicts.append(this_b_n_dict)

        df_block_neuron = pd.DataFrame(block_neuron_dicts)
        df_block_neuron.to_pickle(f"{output_dir}/all_blocks_{df_name}_phoneme_activations.pkl")
        df_types[df_name] = df_block_neuron.copy()

    # number of top phonemes per neuron per block
    df_types['phoneme2shuffled']['top_phonemes_count'] = df_types['phoneme2shuffled']['top_phonemes'].apply(lambda x: len(x))
    df_types['phoneme2noise']['top_phonemes_count'] = df_types['phoneme2noise']['top_phonemes'].apply(lambda x: len(x))
    df_types['phoneme2shuffled']['bottom_phonemes_count'] = df_types['phoneme2shuffled']['bottom_phonemes'].apply(lambda x: len(x))
    df_types['phoneme2noise']['bottom_phonemes_count'] = df_types['phoneme2noise']['bottom_phonemes'].apply(lambda x: len(x))

    # plot number of top phonemes per neuron per block
    fig,axes = plt.subplots(len(df_types.keys()), len(groups), figsize=(20, 10))#,sharex=True, sharey=True)
    experiment_map = {'phoneme2shuffled':'d value of Phoneme vs Shuffled', 'phoneme2noise':'d value of Phoneme vs AM Noise'}
    for i,dtype in enumerate(df_types.keys()):
        groups=df_types[dtype].groupby('block_index')

        for name, group in groups:
            j = int(name)
            sorted_top = group.sort_values('num_great_cohen', ascending=False)
            sorted_bottom = group.sort_values('bottom_phonemes_count', ascending=False)
            ax = axes[i][j]
            ax.plot(sorted_top['num_worst_cohen'].to_list(), label='# Phonemes with d < -0.8')
            ax.plot(sorted_top['num_great_cohen'].to_list(), label='# Phonemes with d > 0.8')
            ax.hlines(np.mean(
                sorted_top['num_worst_cohen']),xmin=0, xmax=len(sorted_top['num_worst_cohen'])-1
                , color='black', linestyle='--', label='Mean # Phonemes with d < -0.8')
            ax.hlines(np.mean(
                sorted_top['num_great_cohen']),xmin=0, xmax=len(sorted_top['num_great_cohen'])-1
                , color='red', linestyle='--', label='Mean # Phonemes with d > 0.8')

            ax.set_xlabel('Neurons')
            ax.set_ylabel(f'Number of Phonemes per Neuron\n{experiment_map[dtype]}')
            ax.set_title(f'Block {name}')
            ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close('all')
if __name__ == "__main__":
    main()