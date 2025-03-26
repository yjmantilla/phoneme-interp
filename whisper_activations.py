def main():
    import numpy as np
    import whisper
    import torch
    import pickle
    from collections import defaultdict
    from scipy.stats import ttest_ind
    import joblib
    import seaborn as sns
    import pandas as pd
    import os
    import matplotlib.pyplot as plt 
    import argparse

    import argparse

    parser = argparse.ArgumentParser(description="Extract and analyze phoneme activations from Whisper.")

    parser.add_argument(
        "--phoneme_file", 
        type=str, 
        default="phoneme_segments.pkl", 
        help="Path to the pickled phoneme segments file (default: phoneme_segments.pkl)"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output", 
        help="Directory where outputs (activations, plots, etc.) will be saved (default: output)"
    )

    parser.add_argument(
        "--block_index", 
        type=int, 
        default=2, 
        help="Whisper encoder block index to extract MLP activations from (default: 2)"
    )

    # example usage
    # python -u whisper_activations.py --phoneme_file phoneme_segments.pkl --output_dir output --block_index 2
    args = parser.parse_args()

    phoneme_file = args.phoneme_file
    output_dir = args.output_dir
    block_index = args.block_index

    with open(phoneme_file, "rb") as f:
        df = pickle.load(f)

    WHISPER_SAMPLE_RATE = 16000
    WHISPER_INPUT_SAMPLES = 30 * WHISPER_SAMPLE_RATE  # 30 seconds*16k = 480000
    WHISPER_NUM_FRAMES = 1500
    SAMPLES_PER_FRAME = WHISPER_INPUT_SAMPLES // WHISPER_NUM_FRAMES  # 320

    def pad_or_truncate(segment, target_len=WHISPER_INPUT_SAMPLES):
        if len(segment) > target_len:
            return segment[:target_len]
        else:
            return np.pad(segment, (0, target_len - len(segment)))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model to the same device
    model = whisper.load_model("tiny").to(device)  # small, base, large
    # Set model to eval mode
    model.eval()

    # Store activations per segment
    activations_list = []

    # Hook dict
    activation_dict = {}

    def hook_fn(module, input, output):
        activation_dict['mlp'] = output.detach().cpu()  # Move to CPU immediately

    # Register hook to desired block's MLP
    mlp_module = model.encoder.blocks[block_index].mlp
    mlp_string = f'model.encoder.blocks[{block_index}].mlp'
    hook = mlp_module.register_forward_hook(hook_fn)

    mlp_path = f"{output_dir}/{mlp_string}/"
    os.makedirs(mlp_path, exist_ok=True)
    activations_path = f"{mlp_path}/activations.pt"
    tooshort_path = f"{mlp_path}/too_short.npy"

    too_short = []
    if not os.path.exists(activations_path):
        # Process each row
        for i, row in df.iterrows():
            original_segment = np.asarray(row['segment'], dtype=np.float32)
            segment = pad_or_truncate(original_segment)

            # Convert to tensor and move to correct device
            audio_tensor = torch.from_numpy(segment).unsqueeze(0).to(device)  # Shape: (1, T)

            # Compute log-mel (this returns CPU tensor)
            mel = whisper.log_mel_spectrogram(audio_tensor).to(device)  # Move to correct device

            # Forward pass (hook will capture MLP activations)
            with torch.no_grad():
                _ = model.encoder(mel)

            # Store only the relevant slice of activations
            if 'mlp' in activation_dict:
                n_samples = len(original_segment)
                n_frames = n_samples // SAMPLES_PER_FRAME
                if n_frames == 0:
                    print(f"Too short: {len(original_segment)} samples -> 0 frames. Going to at least 1 frames.")
                    n_frames = 1
                    too_short.append(True)
                else:
                    too_short.append(False)
                activation_slice = activation_dict['mlp'][0, :n_frames, :]
                activations_list.append(activation_slice)
            else:
                activations_list.append(None)  # In case hook fails for some reason

            print(f"Processed segment {i}, key: {row['key'].split('_')}")

        # save activations_list

        torch.save(activations_list, activations_path)  # binary, efficient
        print(f"Saved activations to {activations_path}")
        np.save(tooshort_path, np.array(too_short))

    else:
        print(f"Loading activations from {activations_path}")
        activations_list = torch.load(activations_path)
        too_short = np.load(tooshort_path)

    # get mean max min std of activations across frame dimension, as we want to get stats at the neuron level

    foo_dict = {
        'max': lambda x: torch.max(x, dim=0).values,
        'min': lambda x: torch.min(x, dim=0).values,
        'mean': lambda x: torch.mean(x, dim=0),
        'std': lambda x: torch.std(x, dim=0)
    }

    for key, func in foo_dict.items():
        print(f"Computing {key} of activations")
        df[f'activations_{mlp_string}_{key}'] = [func(activations) for activations in activations_list]

    shapes =[]
    for i in range(len(activations_list)):
        shapes.append(activations_list[i].shape)
        

    # Choose the stat you want to analyze (mean, max, etc.)
    stat_key_pattern = f'activations_model.encoder.blocks[{block_index}].mlp_%'

    def get_neuron_vs_phoneme_matrix(df,stat_key):
        df = df.copy()
        # Convert the list-of-arrays column into a matrix
        df['activ_array'] = df[stat_key].apply(lambda x: np.array(x) if x is not None else np.zeros(384))

        # Group by phoneme
        grouped = df.groupby('phoneme')['activ_array']

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

    results_mean = get_neuron_vs_phoneme_matrix(df,stat_key_pattern.replace('%','mean'))
    results_max = get_neuron_vs_phoneme_matrix(df,stat_key_pattern.replace('%','max'))


    #stat_key = stat_key_pattern.replace('%','mean')
    #neuron_idx = 42
    def get_phoneme_vs_phoneme_per_neuron_matrix(df, stat_key, neuron_idx):

        phoneme_groups = defaultdict(list)

        for _, row in df.iterrows():
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
        print(neuron_idx)
        return output_dict


    def get_phoneme_vs_phoneme_all_neurons(df, stat_key,njobs=1):
        phoneme_vs_phoneme_matrix = {}
        num_neurons = df[stat_key].iloc[0].shape[0]

        if njobs == 1:
            for neuron_idx in range(num_neurons):
                print(f"Processing neuron {neuron_idx}")
                phoneme_vs_phoneme_matrix[neuron_idx] = get_phoneme_vs_phoneme_per_neuron_matrix(df, stat_key, neuron_idx)
        else:
            phoneme_vs_phoneme_matrix_ = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(get_phoneme_vs_phoneme_per_neuron_matrix)(df, stat_key, neuron_idx)
                for neuron_idx in range(num_neurons)
            )
            for neuron_idx in range(num_neurons):
                phoneme_vs_phoneme_matrix[neuron_idx] = phoneme_vs_phoneme_matrix_[neuron_idx]
        return phoneme_vs_phoneme_matrix

    if not os.path.exists(f"{mlp_path}/phoneme_vs_phoneme_matrix_mean.pkl"):
        phoneme_vs_phoneme_matrix_mean = get_phoneme_vs_phoneme_all_neurons(df, stat_key.replace('%','mean'),njobs=8)
        with open(f"phoneme_vs_phoneme_matrix_mean.pkl", "wb") as f:
            pickle.dump(phoneme_vs_phoneme_matrix_mean, f)
    else:
        print(f"Loading phoneme_vs_phoneme_matrix_mean")
        with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_mean.pkl", "rb") as f:
            phoneme_vs_phoneme_matrix_mean = pickle.load(f)
    if not os.path.exists(f"{mlp_path}/phoneme_vs_phoneme_matrix_max.pkl"):
        phoneme_vs_phoneme_matrix_max = get_phoneme_vs_phoneme_all_neurons(df, stat_key.replace('%','max'),njobs=8)
        with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_max.pkl", "wb") as f:
            pickle.dump(phoneme_vs_phoneme_matrix_max, f)
    else:
        print(f"Loading phoneme_vs_phoneme_matrix_max")
        with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_max.pkl", "rb") as f:
            phoneme_vs_phoneme_matrix_max = pickle.load(f)


    def plot_phoneme_vs_phoneme_matrix(matrix, neuron_id=None, value_type='Cohen\'s d', center=0, cmap='coolwarm'):
        """
        Plot phoneme vs. phoneme matrix for a neuron.

        Parameters:
            matrix (pd.DataFrame): Square DataFrame (phoneme × phoneme).
            neuron_id (int): Optional neuron index to show in title.
            value_type (str): Label for values (e.g., 't-stat', 'Cohen\'s d').
            center (float): Center value for colormap (e.g., 0 if values are signed).
            cmap (str): Colormap to use (e.g., 'coolwarm', 'viridis').
        """
        fig = plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            matrix.astype(float), 
            cmap=cmap, 
            xticklabels=True, 
            yticklabels=True,
            annot=False, 
            fmt=".2f",
            center=center,
            square=True,
            cbar_kws={"label": value_type}
        )

        title = f"Phoneme Discrimination – Neuron {neuron_id}" if neuron_id is not None else "Phoneme Discrimination Matrix"
        title+=f" ({value_type})"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Phoneme B")
        ax.set_ylabel("Phoneme A")
        plt.tight_layout()
        return fig



    def plot_phoneme_discrimination_boxplot(matrix, neuron_id, matrix_label,key):
        """
        Plot boxplots of how strongly each phoneme is discriminated from others by a neuron.
        """
        # Melt matrix to long format
        long_df = matrix.reset_index().melt(id_vars='index')
        long_df.columns = ['phoneme_a', 'phoneme_b', 'value']

        # Remove self-comparisons (diagonal)
        long_df = long_df[long_df['phoneme_a'] != long_df['phoneme_b']]

        # Sort phonemes by median discrimination strength (for plotting)
        medians = long_df.groupby('phoneme_a')['value'].median().sort_values(ascending=False)
        long_df['phoneme_a'] = pd.Categorical(long_df['phoneme_a'], categories=medians.index, ordered=True)

        # Plot
        fig = plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=long_df, x='phoneme_a', y='value', palette='gist_rainbow', hue='phoneme_a',legend=False)
        ax.set_title(f"Phoneme Discrimination Distribution – Neuron {neuron_id} ({matrix_label} {key})", fontsize=14)
        ax.set_xlabel("Phoneme")
        ax.set_ylabel(f"{matrix_label} vs others")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
        return fig

    #phoneme_vs_phoneme_matrix = phoneme_vs_phoneme_matrix_mean
    #phoneme_vs_phoneme_matrix[0].keys()
    #matrix = phoneme_vs_phoneme_matrix[0]['matrix_p']

    figpath=f'{mlp_path}/figures'
    os.makedirs(figpath, exist_ok=True)
    for phoneme_vs_phoneme_matrix,matrix_label in zip([phoneme_vs_phoneme_matrix_max, phoneme_vs_phoneme_matrix_mean],['max','mean']):
        for neuron_id in phoneme_vs_phoneme_matrix.keys():
            for key in ['matrix_t','matrix_p','matrix_d']:
                matrix = phoneme_vs_phoneme_matrix[neuron_id][key]

                # remove eng phoneme
                matrix = matrix.drop('eng', axis=0, errors='ignore')
                matrix = matrix.drop('eng', axis=1, errors='ignore')
                if key == 'matrix_p':
                    matrix = matrix.map(lambda x: -np.log10(x))
                if key in ['matrix_t','matrix_d']:
                    matrix = matrix.map(lambda x: np.abs(x))
                
                # apply average over the rows
                fig = plot_phoneme_vs_phoneme_matrix(matrix, neuron_id=neuron_id, value_type=' '.join([matrix_label,key]))
                fig.savefig(f"{figpath}/{matrix_label}_{key}_neuron_{neuron_id}.png")
                plt.close(fig)

                fig = plot_phoneme_discrimination_boxplot(matrix, neuron_id=neuron_id, matrix_label=matrix_label,key=key)
                fig.savefig(f"{figpath}/sorted_{matrix_label}_{key}_neuron_{neuron_id}.png")
                plt.close(fig)

    #plot_phoneme_vs_phoneme_matrix(matrix, neuron_id=0, value_type='t-stat', center=0, cmap='coolwarm')

    if False:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        def plot_waveform_with_activation_stats(
            waveform,
            activations,
            phoneme=None,
            utterance=None,
            sample_rate=16000,
            std_alpha=0.2
        ):
            """
            Plot waveform with activation statistics (mean, min, max, std) on a secondary y-axis.

            Parameters:
                waveform (np.ndarray): 1D float32 array of waveform samples.
                activations (torch.Tensor): Shape (n_frames, hidden_dim), MLP activations.
                phoneme (str): Optional label for phoneme.
                utterance (str): Optional label for utterance.
                sample_rate (int): Default 16000 Hz.
                std_alpha (float): Transparency for std shading.
            """
            waveform = np.asarray(waveform)
            n_samples = len(waveform)
            n_frames, hidden_dim = activations.shape

            # Time vectors
            time_waveform = np.linspace(0, n_samples / sample_rate, n_samples)
            frame_times = np.linspace(0, n_samples / sample_rate, n_frames)

            # Aggregate stats over hidden dim
            mean_vals = torch.mean(activations, dim=1).numpy()
            min_vals = torch.min(activations, dim=1).values.numpy()
            max_vals = torch.max(activations, dim=1).values.numpy()
            std_vals = torch.std(activations, dim=1).numpy()

            # Plot
            fig, ax1 = plt.subplots(figsize=(12, 4))

            # Left y-axis: waveform
            ax1.plot(time_waveform, waveform, color='black', linewidth=1, label='Waveform')
            ax1.set_ylabel("Amplitude", color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # Right y-axis: activations
            ax2 = ax1.twinx()
            ax2.plot(frame_times, mean_vals, label='Mean', color='purple', linewidth=2)
            ax2.plot(frame_times, min_vals, label='Min', color='blue', linestyle='dashed')
            ax2.plot(frame_times, max_vals, label='Max', color='red', linestyle='dashed')

            # Shaded std area
            ax2.fill_between(
                frame_times,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color='purple',
                alpha=std_alpha,
                label='±1 STD'
            )

            ax2.set_ylabel("MLP Activation Value", color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Title
            title = f"Waveform + MLP Activation Stats"
            if phoneme:
                title += f" – Phoneme: '{phoneme}'"
            if utterance:
                title += f" – Utterance: {utterance}"
            ax1.set_title(title)

            # Legend
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.show()


        row = df.iloc[42]
        waveform = np.asarray(row['segment'], dtype=np.float32)
        activations = row['activations_block2_mlp']

        # plot_waveform_with_activation_stats(
        #     waveform=waveform,
        #     activations=activations,
        #     phoneme=row.get('phoneme'),
        #     utterance=row.get('utterance')
        # )


if __name__ == "__main__":
    main()