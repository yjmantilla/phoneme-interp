import nltk
from nltk.corpus import timit

# Uncomment the following line if you haven't downloaded TIMIT corpus yet
#nltk.download('timit')

utterances = [x.split('.phn')[0] for x in nltk.corpus.timit.fileids() if x.endswith('.phn')]

for utterance in utterances:

    # Get phonemes and their corresponding time alignments
    phonemes = timit.phones(utterance)
    times = timit.phone_times(utterance)

    # Get the audio data for the utterance
    audio = timit.audiodata(utterance)

    # Create a dictionary to store phoneme segments
    phoneme_segments = {}

    # Extract audio segments for each phoneme
    for (phone,start, end), phoneme in zip(times, phonemes):
        # Convert time to sample indices
        start_sample = int(start * 16000)  # TIMIT uses 16kHz sampling rate
        end_sample = int(end * 16000)
        
        # Extract the audio segment for this phoneme
        segment = audio[start_sample:end_sample]
        
        # Store the segment
        if phone not in phoneme_segments:
            phoneme_segments[phone] = []
        phoneme_segments[phone].append(segment)

    # Print information about the segments
    for phone, segments in phoneme_segments.items():
        print(f"Phoneme '{phone}' has {len(segments)} segments")
