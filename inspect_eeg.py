path = '/home/yorguin/scratch/datasets/ds004504-download/sub-001/eeg/sub-001_task-eyesclosed_eeg.set'

import mne

raw = mne.io.read_raw_eeglab(path, preload=True)

print(raw.info)

# Get the sampling frequency
sfreq = raw.info['sfreq']
print(f'Sampling frequency: {sfreq} Hz')


# Get the number of channels
n_channels = raw.info['nchan']
print(f'Number of channels: {n_channels}')

# Get the channel names
channel_names = raw.info['ch_names']
print(f'Channel names: {channel_names}')


# save a plot of the psd for the first 10 seconds of data
import matplotlib.pyplot as plt
raw.plot_psd(fmax=50, tmin=0, tmax=10)
plt.savefig('psd_plot.png')
