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

# Optional Griffin-Lim (for spectrogram inversion)
try:
    import torchaudio
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400)
except ImportError:
    griffin_lim = None


def play_float_audio(float_array, sample_rate=16000, sample_width=2, channels=1):
    """
    Plays a float32 NumPy array (values in [-1, 1]) as audio.
    """
    int_data = (float_array * 32768.0).astype('<i2')
    pcm_bytes = int_data.tobytes()
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)
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
    parser.add_argument("--output", type=str, default="optimized_output.npy", help="Output .npy file for generated waveform or mel")
    parser.add_argument("--optimize_space", type=str, default="waveform", choices=["waveform", "spectrogram"], help="Whether to optimize waveform or mel-spectrogram directly")

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
    print(f"[INFO] Using device: {device} | Optimization space: {args.optimize_space}")

    # Load Whisper model
    model = whisper.load_model(args.model_size).to(device).eval()

    # Register hook
    activations = {}
    def hook_fn(module, input, output):
        activations["mlp"] = output.detach()

    mlp_module = model.encoder.blocks[args.block_index].mlp
    mlp_module.register_forward_hook(hook_fn)

    sample_rate = 16000
    total_samples = 30 * sample_rate
    opt_len = int(args.optimize_seconds * sample_rate)
    pad_len = total_samples - opt_len

    if args.optimize_space == "waveform":
        x_opt = torch.randn(1, opt_len, device=device) * 0.01
        x_opt.requires_grad_()
        optimizer = torch.optim.Adam([x_opt], lr=args.lr)

        for step in range(args.steps):
            optimizer.zero_grad()
            x_full = torch.cat([x_opt, torch.zeros(1, pad_len, device=device)], dim=1)
            mel = whisper.log_mel_spectrogram(x_full).to(device)
            _ = model.encoder(mel)
            neuron_act = activations["mlp"][0, :, args.neuron_index]

            # Choose loss
            if args.loss_type == "mean":
                loss = neuron_act.mean()
            elif args.loss_type == "max":
                loss = neuron_act.max()
            elif args.loss_type == "quantile":
                loss = torch.quantile(neuron_act, args.quantile)

            loss -= args.l2_decay * (x_opt ** 2).mean()
            (-loss).backward()
            optimizer.step()

            if args.blur_sigma > 0:
                x_opt.data = blur(x_opt.data, args.blur_sigma)

            if step % 20 == 0:
                print(f"Step {step:4d} | Activation = {loss.item():.4f}")

        result = torch.cat([x_opt, torch.zeros(1, pad_len, device=device)], dim=1).detach().cpu().squeeze().numpy()
        np.save(args.output, result)
        print(f"[INFO] Saved optimized waveform to {args.output}")

        plt.figure(figsize=(12, 3))
        plt.plot(result)
        plt.title(f"Maximized Neuron {args.neuron_index} (Block {args.block_index})")
        plt.xlabel("Sample Index")
        plt.tight_layout()
        plt.show()

        play_float_audio(result, sample_rate=sample_rate)

        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((result * 32767).astype(np.int16).tobytes())
        print("[INFO] Saved waveform to output.wav")

    elif args.optimize_space == "spectrogram":
        # Whisper spectrograms are (1, 80, 1500) for 30 seconds of audio
        full_frames = 1500
        opt_frames = int(args.optimize_seconds * 50)  # 50 frames per second
        pad_frames = full_frames - opt_frames

        # Optimize only the first N frames
        mel_opt_part = torch.randn(1, 80, opt_frames, device=device) * 0.01
        mel_opt_part.requires_grad_()

        optimizer = torch.optim.Adam([mel_opt_part], lr=args.lr)

        for step in range(args.steps):
            optimizer.zero_grad()

            # Concatenate optimized + padding (non-trainable)
            mel_full = torch.cat([mel_opt_part, torch.zeros(1, 80, pad_frames, device=device)], dim=2)

            _ = model.encoder(mel_full)
            neuron_act = activations["mlp"][0, :, args.neuron_index]

            # Choose loss
            if args.loss_type == "mean":
                loss = neuron_act.mean()
            elif args.loss_type == "max":
                loss = neuron_act.max()
            elif args.loss_type == "quantile":
                loss = torch.quantile(neuron_act, args.quantile)

            loss -= args.l2_decay * (mel_opt_part ** 2).mean()
            (-loss).backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"Step {step:4d} | Activation = {loss.item():.4f}")

        mel_final = mel_full.detach().cpu().squeeze().numpy().T  # shape (1500, 80)

        np.save(args.output, mel_final)
        print(f"[INFO] Saved optimized mel spectrogram to {args.output}")

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_final.T, aspect='auto', origin='lower')
        plt.title(f"Mel Spectrogram – Neuron {args.neuron_index}")
        plt.colorbar(label="Amplitude")
        plt.tight_layout()
        plt.show()

        # Try to invert to waveform if torchaudio is available
        if griffin_lim is not None:
            print("[INFO] Attempting to invert mel spectrogram to audio...")
            mel_tensor = torch.tensor(mel_final.T).unsqueeze(0)
            inv_waveform = griffin_lim(mel_tensor).squeeze().numpy()
            play_float_audio(inv_waveform)
            with wave.open("output_from_mel.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((inv_waveform * 32767).astype(np.int16).tobytes())
            print("[INFO] Saved inverted waveform to output_from_mel.wav")
        else:
            print("[WARN] torchaudio not found. Skipping waveform inversion.")

if __name__ == "__main__":
    main()
