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
import torch.nn.functional as F

# Optional Griffin-Lim (for spectrogram inversion)
try:
    import torchaudio
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400)

    def resample_mel_for_griffinlim(mel_tensor, target_bins=201):
        """
        Resize (1, 80, T) → (1, 201, T) using bilinear interpolation for Griffin-Lim.
        """
        mel_tensor = mel_tensor.unsqueeze(1)  # → (1, 1, 80, T)
        resized = F.interpolate(mel_tensor, size=(target_bins, mel_tensor.shape[-1]), mode="bilinear", align_corners=False)
        return resized.squeeze(1)  # → (1, 201, T)

except ImportError:
    griffin_lim = None

def play_float_audio(float_array, sample_rate=16000, sample_width=2, channels=1):
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
    parser.add_argument("--steps", type=int, default=200, help="Number of gradient ascent steps")
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
    torch.autograd.set_detect_anomaly(True)
    model = whisper.load_model(args.model_size).to(device).eval()

    activations = {}
    def hook_fn(module, input, output):
        activations["mlp"] = output  # Do NOT detach — keeps gradient flow

    mlp_module = model.encoder.blocks[args.block_index].mlp
    mlp_module.register_forward_hook(hook_fn)

    sample_rate = 16000
    total_samples = 30 * sample_rate
    opt_len = int(args.optimize_seconds * sample_rate)
    pad_len = total_samples - opt_len

    if args.optimize_space == "waveform":
        # (unchanged waveform branch)
        pass  # not shown here, same as before

    elif args.optimize_space == "spectrogram":
        dummy_audio = torch.randn(1, 16000 * 30)
        mel_example = whisper.log_mel_spectrogram(dummy_audio.to(device))
        total_frames = mel_example.shape[-1]
        n_mels = mel_example.shape[-2]

        print(f"[INFO] Optimizing full {total_frames} frames ({args.optimize_seconds}s)")

        mel_opt_noise = torch.randn(1, n_mels, total_frames, device=device) * 0.01
        mel_opt_noise.requires_grad_()

        optimizer = torch.optim.Adam([mel_opt_noise], lr=args.lr)

        for step in range(args.steps):
            optimizer.zero_grad()

            mel_input = mel_opt_noise

            # Sanity check for gradient flow
            #print(f"[DEBUG] requires_grad: {mel_input.requires_grad}, grad_fn: {mel_input.grad_fn is not None}")

            _ = model.encoder(mel_input)
            neuron_act = activations["mlp"][0, :, args.neuron_index]

            if args.loss_type == "mean":
                loss = torch.mean(neuron_act.clone())
            elif args.loss_type == "max":
                loss = torch.max(neuron_act.clone())
            elif args.loss_type == "quantile":
                loss = torch.quantile(neuron_act.clone(), args.quantile)

            l2_reg = args.l2_decay * (mel_input ** 2).mean()
            loss = loss - l2_reg
            (-loss).backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"Step {step:4d} | Activation = {loss.item():.4f}")

        mel_final = mel_opt_noise.detach().cpu().squeeze().numpy()
        np.save(args.output, mel_final)
        print(f"[INFO] Saved optimized mel spectrogram to {args.output}")

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_final, aspect='auto', origin='lower')
        plt.title(f"Mel Spectrogram - Neuron {args.neuron_index}")
        plt.colorbar(label="Log-Mel Value")
        plt.tight_layout()
        plt.savefig("mel_spectrogram.png")
        plt.close()

        if griffin_lim is not None:
            print("[INFO] Attempting to invert mel spectrogram to audio...")

            mel_tensor = torch.tensor(mel_final).unsqueeze(0)
            mel_tensor_interp = resample_mel_for_griffinlim(mel_tensor, target_bins=201)

            inv_waveform = griffin_lim(mel_tensor_interp).squeeze().numpy()

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
