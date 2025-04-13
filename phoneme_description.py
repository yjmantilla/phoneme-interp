

import matplotlib.pyplot as plt
import pandas as pd

# Load the phoneme count CSV
df = pd.read_csv("output\model.encoder.blocks[3].mlp\phoneme_counts.csv")

# Display the top rows to understand its structure
df.head()


# Filter out original phonemes (exclude those with "@shuffled" or "@noise")
df_original = df[~df['phoneme'].str.contains("@")].copy()

# Sort by count
df_original = df_original.sort_values("count", ascending=False)

# reset index to get the original phoneme order
df_original.reset_index(drop=True, inplace=True)

# Identify the cutoff index: the first phoneme with count >= 10
cutoff_index = df_original[df_original["count"] < 10].index[0] if not df_original[df_original["count"] < 10].empty else len(df_original)

# Plotting again with a vertical line
plt.figure(figsize=(10, 4))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

bars = plt.bar(df_original["phoneme"], df_original["count"], edgecolor='black')
plt.xticks(rotation=90)
plt.title("Phoneme Distribution in TIMIT Subset")
plt.xlabel("Phoneme")
plt.ylabel("Count")
plt.grid(True, linestyle='--', linewidth=0.5, axis='y')

# Add vertical line to indicate excluded phonemes
plt.axvline(x=cutoff_index - 0.5, color='red', linestyle='--', label='Exclusion Threshold (<10 samples)')
plt.legend()

plt.tight_layout()
plt.savefig("phoneme_distribution.pdf", bbox_inches='tight')
plt.savefig("phoneme_distribution.png", bbox_inches='tight')
plt.show()


dfpath = r"Y:\code\phoneme-interp\output\model.encoder.blocks[0].mlp\phoneme_activations.pkl"

df = pd.read_pickle(dfpath)

segment = df.iloc[0]['segment']
from scipy.signal import hilbert
import numpy as np

WHISPER_SAMPLE_RATE = 16000
WHISPER_INPUT_SAMPLES = 30 * WHISPER_SAMPLE_RATE  # 30 seconds*16k = 480000
WHISPER_NUM_FRAMES = 1500
SAMPLES_PER_FRAME = WHISPER_INPUT_SAMPLES // WHISPER_NUM_FRAMES  # 320

def smooth_envelope(x):
    analytic = hilbert(x)
    amplitude_envelope = np.abs(analytic)
    return amplitude_envelope / np.max(amplitude_envelope)

def generate_am_noise(seg, seed=42):
    np.random.seed(seed)
    noise = np.random.randn(len(seg))
    envelope = smooth_envelope(seg)
    return (noise * envelope* np.max(np.abs(seg))).astype(np.float32)


original_segment=df.iloc[0]['segment']
original_noise = generate_am_noise(original_segment)
original_shuffling = np.random.permutation(original_segment)


import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

def compute_fft(x, sr=16000):
    N = len(x)
    yf = np.abs(fft(x))[:N//2]
    xf = fftfreq(N, 1/sr)[:N//2]
    return xf, yf

# List of segments and titles
segments = [original_segment, original_noise, original_shuffling]
titles = ['Original','AM Noise', 'Shuffled']

# Plot setup
fig, axs = plt.subplots(3, 2, figsize=(12, 6))

for i, seg in enumerate(segments):
    # Time domain
    axs[i, 0].plot(seg, color='black')
    axs[i, 0].set_title(f"{titles[i]} - Time Domain", fontsize=10)
    axs[i, 0].set_xlabel("Sample", fontsize=9)
    axs[i, 0].set_ylabel("Amplitude", fontsize=9)
    axs[i, 0].tick_params(labelsize=8)

    # Frequency domain
    xf, yf = compute_fft(seg)
    axs[i, 1].plot(xf, yf, color='black')
    axs[i, 1].set_title(f"{titles[i]} - Frequency Domain", fontsize=10)
    axs[i, 1].set_xlabel("Frequency (Hz)", fontsize=9)
    axs[i, 1].set_ylabel("Magnitude", fontsize=9)
    axs[i, 1].tick_params(labelsize=8)

# Global style
plt.tight_layout()
plt.savefig("control_conditions_example.pdf", dpi=300, bbox_inches='tight')
plt.savefig("control_conditions_example.png", dpi=300, bbox_inches='tight')
plt.show()

