import argparse
import torch
import whisper
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pygame.mixer
import wave
from io import BytesIO
import time

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


# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gradient ascent on Whisper to maximize neuron activation.")
    
    parser.add_argument("--model_size", type=str, default="tiny", help="Whisper model size (tiny, base, small, etc.)")
    parser.add_argument("--block_index", type=int, default=2, help="Encoder block index to target")
    parser.add_argument("--neuron_index", type=int, default=1, help="Neuron index within the MLP layer")
    parser.add_argument("--steps", type=int, default=20000, help="Number of gradient ascent steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--l2_decay", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur sigma for regularization (0 = off)")
    parser.add_argument("--output", type=str, default="optimized_waveform.npy", help="Output .npy file for generated waveform")

    return parser.parse_args()

# -------------------------
# Blur helper
# -------------------------
def blur(signal, sigma):
    return torch.tensor(gaussian_filter1d(signal.detach().cpu().numpy(), sigma)).to(signal.device)

# -------------------------
# Main gradient ascent
# -------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load Whisper model
    model = whisper.load_model(args.model_size).to(device).eval()

    # Register hook
    activations = {}
    def hook_fn(module, input, output):
        activations["mlp"] = output.detach()

    mlp_module = model.encoder.blocks[args.block_index].mlp
    mlp_module.register_forward_hook(hook_fn)

    # Start from random noise waveform
    target_len = 480000  # 30s * 16kHz
    x = torch.randn(1, target_len, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([x], lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad()

        mel = whisper.log_mel_spectrogram(x).to(device)
        _ = model.encoder(mel)

        act = activations["mlp"]  # (1, frames, dim)
        neuron_act = act[0, :, args.neuron_index]
        loss = neuron_act.mean()

        # L2 regularization
        loss -= args.l2_decay * (x ** 2).mean()

        (-loss).backward()
        optimizer.step()

        if args.blur_sigma > 0:
            x.data = blur(x.data, args.blur_sigma)

        if step % 20 == 0:
            print(f"Step {step:3d} | Activation = {loss.item():.4f}")

    result = x.detach().cpu().squeeze().numpy()
    np.save(args.output, result)
    print(f"[INFO] Saved optimized waveform to {args.output}")

    # Optional plot
    plt.figure(figsize=(12, 3))
    plt.plot(result)
    plt.title(f"Maximized Neuron {args.neuron_index} (Block {args.block_index})")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

    # play the audio

    play_float_audio(result, sample_rate=16000)

    pygame.mixer.init(frequency=16000, size=-16, channels=1)
    byte_io = BytesIO()
    wav_file = wave.open(byte_io, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)  # 16-bit audio
    wav_file.setframerate(16000)
    wav_file.writeframes((result * 32767).astype(np.int16).tobytes())
    wav_file.close()
    


if __name__ == "__main__":
    main()
