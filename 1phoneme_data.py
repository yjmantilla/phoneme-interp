#use arena environment
import nltk
from nltk.corpus import timit
import numpy as np
import pygame.mixer
import numpy as np
import wave
from io import BytesIO
import time

# Uncomment the following line if you haven't downloaded TIMIT corpus yet
#nltk.download('timit')

# Create a dictionary to store phoneme segments across all utterances
phoneme_segments = {}
phoneme_counts = {}

utterances = timit.utteranceids()#[x.split('.phn')[0] for x in nltk.corpus.timit.fileids() if x.endswith('.phn')]

def decode_segment(raw_bytes):
    """Convert 16-bit PCM bytes to float32 normalized array."""
    int_data = np.frombuffer(raw_bytes, dtype='<i2')  # Little-endian int16
    return int_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

def play(timit,utterance, start, end):

    pygame.mixer.init(frequency=16000)

    # Assuming self.wav(...) returns raw WAV bytes
    wav_bytes = timit.wav(utterance, start, end)

    # Use BytesIO for binary data
    f = BytesIO(wav_bytes)

    # Load and play
    sound = pygame.mixer.Sound(file=f)
    sound.play()

    # Wait for playback to finish
    while pygame.mixer.get_busy():
        time.sleep(0.01)


def play_pcm_segment(pcm_bytes, sample_rate=16000, sample_width=2, channels=1):
    """
    Takes raw PCM bytes and plays them as audio by wrapping into a WAV container.
    """

    buffer = BytesIO()

    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    buffer.seek(0)

    # Init mixer and play
    pygame.mixer.init(frequency=sample_rate)
    sound = pygame.mixer.Sound(file=buffer)
    sound.play()

    while pygame.mixer.get_busy():
        time.sleep(0.01)

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

# Iterate over all fileids (utterances) in the corpus
for utterance in utterances:
    try:
        # Get phonemes and their corresponding time alignments for this utterance
        phonemes = timit.phones(utterance)
        times = timit.phone_times(utterance)
        
        # Get the audio data for the utterance        
        # Extract audio segments for each phoneme in this utterance
        for (phone, start, end), phoneme in zip(times, phonemes):

            if phone not in phoneme_counts:
                phoneme_counts[phone] = 0
            
            i=phoneme_counts[phone]

            key = f"{phone}_{utterance}_{i}"
            # Extract the audio segment for this phoneme
            segment = decode_segment(timit.audiodata(utterance,start=start,end=end))
            #play(timit,utterance, start, end)
            #play_pcm_segment(timit.audiodata(utterance,start=start,end=end))
            #play_float_audio(segment)
            phoneme_segments[key] = {}
            phoneme_segments[key]['segment'] = segment
            phoneme_segments[key]['phoneme'] = phone
            phoneme_segments[key]['utterance'] = utterance
            phoneme_segments[key]['within_phone_count'] = i
            
            phoneme_counts[phone] = i+1
        print(f"Processed utterance: {utterance}")
        
    except Exception as e:
        print(f"Error processing utterance {utterance}: {str(e)}")



import pandas as pd
df = pd.DataFrame.from_dict(phoneme_segments, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index': 'key'}, inplace=True)

# for i in range(10):
#     play_float_audio(df.iloc[i]['segment'])
#     time.sleep(1)


# save pandas dataframe to pickle
df.to_pickle("phoneme_segments.pkl")
