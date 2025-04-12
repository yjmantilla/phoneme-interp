# example usage
# use arena environment
# python -u whisper_activations.py --phoneme_file phoneme_segments.pkl --output_dir output --block_index 2 >  activations_block_2.log

def main():
    import numpy as np
    import whisper
    import torch
    import pickle
    import pandas as pd
    import os
    import argparse
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
    n_frames_path = f"{mlp_path}/n_frames.npy"
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


    too_short = []
    n_frames_list = []
    # i=0;row = df.iloc[0];
    if not os.path.exists(segment_act_path):
        # Process each row
        for i, row in df.iterrows():
            original_segment = np.asarray(row['segment'], dtype=np.float32)
            # Add noise
            # segment = original_segment + generate_am_noise(original_segment)
            original_noise = generate_am_noise(original_segment)
            original_shuffling = np.random.permutation(original_segment)

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
            n_frames_list.append(n_frames)

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
        np.save(n_frames_path, np.array(n_frames_list))
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
        'list': lambda x: x,
        'max': lambda x: np.max(x, axis=0),
        'min': lambda x: np.min(x, axis=0),
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'median': lambda x: np.median(x, axis=0),
        'quantile_0.25': lambda x: np.quantile(x, 0.25, axis=0),
        'quantile_0.75': lambda x: np.quantile(x, 0.75, axis=0),
        'quantile_0.9': lambda x: np.quantile(x, 0.9, axis=0),
    }

    df_orig = df.copy()
    df['too_short'] = too_short
    df['n_frames'] = n_frames_list
    df_noise = df.copy()
    df_shuffling = df.copy()
    df_noise['segment'] = noise_list
    df_shuffling['segment'] = shuffling_list
    df_noise['phoneme'] = df_noise['phoneme'] + '@noise'
    df_shuffling['phoneme'] = df_shuffling['phoneme'] + '@shuffled'
    df_noise['key'] = df_noise['phoneme'] + '_' + df_noise['utterance'] + '_' + df_noise['within_phone_count'].astype(str)
    df_shuffling['key'] = df_shuffling['phoneme'] + '_' + df_shuffling['utterance'] + '_' + df_shuffling['within_phone_count'].astype(str)
    df = pd.concat([df,df_noise,df_shuffling],ignore_index=True)

    all_activations = seg_activations_list+noi_activations_list+shu_activations_list # same order as concat
    for key, func in foo_dict.items():
        print(f"Computing {key} of activations")
        df[f'activations_{mlp_string}_{key}'] = [func(activations) for activations in all_activations]

    # save all activations to pickle
    all_activations_path = f"{mlp_path}/all_activations.npy"
    if not os.path.exists(all_activations_path):
        np.save(all_activations_path, np.array(all_activations, dtype=object), allow_pickle=True)
        print(f"Saved all activations to {all_activations_path}")
    
    df.to_pickle(f"{mlp_path}/phoneme_activations.pkl")
    print(f"Saved phoneme activations to {mlp_path}/phoneme_activations.pkl")


if __name__ == "__main__":
    main()