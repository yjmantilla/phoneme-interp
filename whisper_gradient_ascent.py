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
    int_data = (float_array * 32768.0).astype('<i2')

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
    parser.add_argument("--steps", type=int, default=2000, help="Number of gradient ascent steps")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--l2_decay", type=float, default=0, help="L2 regularization strength")
    parser.add_argument("--blur_sigma", type=float, default=0, help="Gaussian blur sigma for regularization (0 = off)")
    parser.add_argument("--optimize_seconds", type=float, default=1.0, help="Number of seconds to optimize at the start of the waveform")
    parser.add_argument("--loss_type", type=str, default="max", choices=["mean", "max", "quantile"], help="Loss type for activation (mean, max, quantile)")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile to use if loss_type=quantile (e.g., 0.9)")
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

    # Determine optimized segment length
    sample_rate = 16000
    total_samples = 30 * sample_rate  # Whisper expects 30s (480000 samples)
    opt_len = int(args.optimize_seconds * sample_rate)
    pad_len = total_samples - opt_len

    # Start from random noise waveform for the optimized region
    #x_opt = torch.randn(1, opt_len, requires_grad=True, device=device)
    # initialize wit low-amplitude noise
    x_opt = torch.randn(1, opt_len, device=device) * 0.01
    x_opt.requires_grad_()  # Makes it a leaf and requires gradient

    optimizer = torch.optim.Adam([x_opt], lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad()

        # Concatenate optimized segment + silence
        x_full = torch.cat([x_opt, torch.zeros(1, pad_len, device=device)], dim=1)

        # Forward pass
        mel = whisper.log_mel_spectrogram(x_full).to(device)
        _ = model.encoder(mel)
        neuron_act = activations["mlp"][0, :, args.neuron_index]

        # Choose loss based on type
        if args.loss_type == "mean":
            loss = neuron_act.mean()
        elif args.loss_type == "max":
            loss = neuron_act.max()
        elif args.loss_type == "quantile":
            loss = torch.quantile(neuron_act, args.quantile)
        else:
            raise ValueError("Invalid loss_type")

        # L2 regularization
        loss -= args.l2_decay * (x_opt ** 2).mean()

        (-loss).backward()
        optimizer.step()

        if args.blur_sigma > 0:
            x_opt.data = blur(x_opt.data, args.blur_sigma)

        if step % 20 == 0:
            print(f"Step {step:4d} | Activation = {loss.item():.4f}")

    # Reconstruct the full waveform: optimized segment + silence
    result = torch.cat([x_opt, torch.zeros(1, pad_len, device=device)], dim=1)
    result = result.detach().cpu().squeeze().numpy()

    np.save(args.output, result)
    print(f"[INFO] Saved optimized waveform to {args.output}")

    # Optional plot
    plt.figure(figsize=(12, 3))
    plt.plot(result)
    plt.title(f"Maximized Neuron {args.neuron_index} (Block {args.block_index})")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

    # Play the audio
    play_float_audio(result, sample_rate=sample_rate)

    # save waveform to wav file

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(result.tobytes())

    print("[INFO] Saved waveform to output.wav")
    

if __name__ == "__main__":
    main()
