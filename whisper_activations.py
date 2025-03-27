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
    import time
    import pygame.mixer
    import wave
    from io import BytesIO
    from scipy.signal import hilbert

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

    # df['phoneme'].value_counts().plot(kind='bar')
    # plt.show()
    df['phoneme'].unique()
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
    seg_activations_list = []
    noi_activations_list = []
    shu_activations_list = []
    segment_list = []
    noise_list = []
    shuffling_list = []

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
    segment_act_path = f"{mlp_path}/segment_activations.npy"
    noise_act_path = f"{mlp_path}/noise_activations.npy"
    shuf_act_path = f"{mlp_path}/shuffling_activations.npy"
    segment_path = f"{mlp_path}/segments.npy"
    noise_path = f"{mlp_path}/noises.npy"
    shuffling_path = f"{mlp_path}/shufflings.npy"

    tooshort_path = f"{mlp_path}/too_short.npy"
    def smooth_envelope(x):
        analytic = hilbert(x)
        amplitude_envelope = np.abs(analytic)
        return amplitude_envelope / np.max(amplitude_envelope)

    def generate_am_noise(seg, seed=42):
        np.random.seed(seed)
        noise = np.random.randn(len(seg))
        envelope = smooth_envelope(seg)
        return (noise * envelope* np.max(np.abs(seg))).astype(np.float32)

    # another option

    def phase_scramble(signal, seed=42):
        np.random.seed(seed)
        fft = np.fft.fft(signal)
        mag = np.abs(fft)
        phase = np.angle(fft)
        
        random_phase = np.random.uniform(-np.pi, np.pi, len(phase))
        new_fft = mag * np.exp(1j * random_phase)
        
        scrambled = np.fft.ifft(new_fft).real
        return scrambled.astype(np.float32)

    def play_float_audio(float_array, sample_rate=16000, sample_width=2, channels=1):
        """
        Plays a float32 NumPy array (values in [-1, 1]) as audio.
        """
        # Step 1: Convert float32 [-1, 1] → int16
        int_data = (float_array * 32768.0).astype('<i2')  # Little-endian int16

        # Step 2: Convert to raw bytes
        pcm_bytes = int_data.tobytes()

        # Step 3: Write to in-memory WAV
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        buffer.seek(0)

        # Step 4: Play
        pygame.mixer.init(frequency=sample_rate)
        sound = pygame.mixer.Sound(file=buffer)
        sound.play()

        while pygame.mixer.get_busy():
            time.sleep(0.01)

    too_short = []
    # i=0;row = df.iloc[0];
    if not os.path.exists(segment_act_path):
        # Process each row
        for i, row in df.iterrows():
            original_segment = np.asarray(row['segment'], dtype=np.float32)
            # Add noise
            # segment = original_segment + generate_am_noise(original_segment)
            original_noise = generate_am_noise(original_segment)
            original_shuffling = np.random.permutation(original_segment)

            # play_float_audio(original_segment)
            # play_float_audio(noise)
            # play_float_audio(shuffling)
            # plt.plot(shuffling)
            # plt.plot(original_segment)
            # plt.plot(noise)
            # plt.show()
            # plt.close()
            segment = pad_or_truncate(original_segment)
            shuffling = pad_or_truncate(original_shuffling)
            noise = pad_or_truncate(original_noise)
            segment_list.append(original_segment)
            noise_list.append(original_noise)
            shuffling_list.append(original_shuffling)

            n_samples = len(original_segment)
            n_frames = n_samples // SAMPLES_PER_FRAME
            if n_frames == 0:
                too_short.append(True)
            else:
                too_short.append(False)

            # Convert to tensor and move to correct device
            audio_tensor = torch.from_numpy(segment).unsqueeze(0).to(device)  # Shape: (1, T)
            # shuffling version
            audio_tensor_shuffling = torch.from_numpy(shuffling).unsqueeze(0).to(device)  # Shape: (1, T)
            # am noise version
            audio_tensor_noise = torch.from_numpy(noise).unsqueeze(0).to(device)

            # Forward pass (hook will capture MLP activations)

            for version_,tensor in zip(['phoneme','shuffled','am-noise'],[audio_tensor, audio_tensor_shuffling, audio_tensor_noise]):
                # Compute log-mel (this returns CPU tensor)
                mel = whisper.log_mel_spectrogram(tensor).to(device)  # Move to correct device

                with torch.no_grad():
                    _ = model.encoder(mel)

                if version_ == 'phoneme':
                    the_list = seg_activations_list
                if version_ == 'shuffled':
                    the_list = shu_activations_list
                if version_ == 'am-noise':
                    the_list = noi_activations_list

                # Store only the relevant slice of activations
                if 'mlp' in activation_dict:
                    n_samples = len(original_segment)
                    n_frames = n_samples // SAMPLES_PER_FRAME
                    if n_frames == 0:
                        print(f"Too short: {len(original_segment)} samples -> 0 frames. Going to at least 1 frames.")
                        n_frames = 1
                    activation_slice = activation_dict['mlp'][0, :n_frames, :]
                    # convert to numpy float 16 to save space
                    activation_slice = activation_slice.numpy().astype(np.float16)
                    the_list.append(activation_slice)
                else:
                    the_list.append(None)  # In case hook fails for some reason

                print(f"Processed segment {i}, key: {row['key'].split('_')}")

        # save activations_list
        assert len(segment_list) == len(seg_activations_list) == len(noise_list) == len(noi_activations_list) == len(shuffling_list) == len(shu_activations_list)
        np.save(noise_act_path, np.array(noi_activations_list,dtype=object),allow_pickle=True)
        np.save(shuf_act_path, np.array(shu_activations_list,dtype=object),allow_pickle=True)
        np.save(segment_act_path, np.array(seg_activations_list,dtype=object),allow_pickle=True)
        np.save(segment_path, np.array(segment_list,dtype=object))
        np.save(noise_path, np.array(noise_list,dtype=object))
        np.save(shuffling_path, np.array(shuffling_list,dtype=object))
        np.save(tooshort_path, np.array(too_short))
        print(f"Saved activations to {segment_act_path}")


    print(f"Loading activations from {segment_act_path}")
    seg_activations_list = np.load(segment_act_path, allow_pickle=True).tolist()
    print(f"Loading activations from {noise_act_path}")
    noi_activations_list = np.load(noise_act_path, allow_pickle=True).tolist()
    print(f"Loading activations from {shuf_act_path}")
    shu_activations_list = np.load(shuf_act_path, allow_pickle=True).tolist()
    print(f"Loading from {segment_path}")
    segment_list = np.load(segment_path, allow_pickle=True)
    print(f"Loading from {noise_path}")
    noise_list = np.load(noise_path, allow_pickle=True)
    print(f"Loading from {shuffling_path}")
    shuffling_list = np.load(shuffling_path, allow_pickle=True)
    print(f"Loading from {tooshort_path}")
    too_short = np.load(tooshort_path, allow_pickle=True)
    # get mean max min std of activations across frame dimension, as we want to get stats at the neuron level

    foo_dict = {
        'max': lambda x: np.max(x, axis=0),
        'min': lambda x: np.min(x, axis=0),
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0)
    }

    df_noise = df.copy()
    df_shuffling = df.copy()
    df_noise['segment'] = noise_list
    df_shuffling['segment'] = shuffling_list
    df_noise['phoneme'] = df_noise['phoneme'] + '@noise'
    df_shuffling['phoneme'] = df_shuffling['phoneme'] + '@shuffled'
    df_noise['key'] = df_noise['phoneme'] + '_' + df_noise['utterance'] + '_' + df_noise['within_phone_count'].astype(str)
    df_shuffling['key'] = df_shuffling['phoneme'] + '_' + df_shuffling['utterance'] + '_' + df_shuffling['within_phone_count'].astype(str)
    df_orig = df.copy()
    df = pd.concat([df,df_noise,df_shuffling],ignore_index=True)

    all_activations = seg_activations_list+noi_activations_list+shu_activations_list # same order as concat
    for key, func in foo_dict.items():
        print(f"Computing {key} of activations")
        df[f'activations_{mlp_string}_{key}'] = [func(activations) for activations in all_activations]

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
        phoneme_vs_phoneme_matrix_mean = get_phoneme_vs_phoneme_all_neurons(df, stat_key_pattern.replace('%','mean'),njobs=12)
        with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_mean.pkl", "wb") as f:
            pickle.dump(phoneme_vs_phoneme_matrix_mean, f)
    if not os.path.exists(f"{mlp_path}/phoneme_vs_phoneme_matrix_max.pkl"):
        phoneme_vs_phoneme_matrix_max = get_phoneme_vs_phoneme_all_neurons(df, stat_key_pattern.replace('%','max'),njobs=12)
        with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_max.pkl", "wb") as f:
            pickle.dump(phoneme_vs_phoneme_matrix_max, f)

    print(f"Loading phoneme_vs_phoneme_matrix_mean")
    with open(f"{mlp_path}/phoneme_vs_phoneme_matrix_mean.pkl", "rb") as f:
        phoneme_vs_phoneme_matrix_mean = pickle.load(f)
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
        fig = plt.figure(figsize=(20, 20))
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
        fig = plt.figure(figsize=(28, 12))
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
                # matrix = matrix.drop('eng', axis=0, errors='ignore')
                # matrix = matrix.drop('eng', axis=1, errors='ignore')
                if key == 'matrix_p':
                    matrix = matrix.map(lambda x: -np.log10(x))
                if key in ['matrix_t','matrix_d']:
                    matrix = matrix.map(lambda x: np.abs(x))
                
                # apply average over the rows
                # fig = plot_phoneme_vs_phoneme_matrix(matrix, neuron_id=neuron_id, value_type=' '.join([matrix_label,key]))
                # fig.savefig(f"{figpath}/{matrix_label}_{key}_neuron_{neuron_id}.png")
                # plt.close(fig)

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