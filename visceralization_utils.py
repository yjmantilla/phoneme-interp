import time
import pygame.mixer
import wave
from io import BytesIO


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
