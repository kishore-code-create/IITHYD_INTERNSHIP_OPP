import os
import sys
import yaml
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

from model import FastSpeech2
from text_utils import text_to_phonemes, phonemes_to_ids, ALL_SYMBOLS

TEST_SENTENCES = [
    {
        "id": "E1",
        "text": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "is_verb": False,
    },
    {
        "id": "E2_a",
        "text": "The orchestra's conduct during the ceremony was praised by the audience.",
        "is_verb": False,
    },
    {
        "id": "E2_b",
        "text": "The researchers will conduct an experiment to evaluate the proposed method.",
        "is_verb": True,
    },
    {
        "id": "E3",
        "text": "The rhyth  m of the orchestra changed unexpectedly.",
        "is_verb": False,
    },
    {
        "id": "E4",
        "text": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
        "is_verb": False,
    },
]

def mel_to_audio(mel_tensor, sample_rate=16000):
    """Inverts 80-bin mel spectrogram to 16kHz audio waveform."""
    mel_exp = torch.exp(mel_tensor.transpose(0, 1)).unsqueeze(0)  # [1, 80, T]
    # Linear spectrogram approximation for Griffin-Lim
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80, sample_rate=sample_rate, f_min=0.0, f_max=8000.0)
    spec = inverse_mel(mel_exp)
    griffin = torchaudio.transforms.GriffinLim(n_fft=1024, win_length=1024, hop_length=256, n_iter=32)
    wav = griffin(spec).squeeze(0).cpu().numpy()
    return wav

def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSpeech2(vocab_size=len(ALL_SYMBOLS), n_speakers=config["model"]["speaker_embedding"]["n_speakers"]).to(device)

    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        print("Warning: Checkpoint not found. Running inference with initialized model.")

    model.eval()

    results_dir = os.path.join(base_dir, "..", "Results", "English")
    audio_dir = os.path.join(results_dir, "test_generated_audios")
    dur_dir = os.path.join(results_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(results_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(results_dir, "variance_adaptor_plots", "energy")

    for d in [audio_dir, dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    print("\nGenerating speech and variance adaptor plots for all English test sentences...")

    with torch.no_grad():
        for item in TEST_SENTENCES:
            sid = item["id"]
            raw_text = item["text"]
            is_verb = item["is_verb"]

            phonemes = text_to_phonemes(raw_text, is_verb_conduct=is_verb)
            token_ids = phonemes_to_ids(phonemes)
            text_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            speaker_tensor = torch.tensor([0], dtype=torch.long, device=device)

            outputs = model(text=text_tensor, speaker_id=speaker_tensor)
            mel_post = outputs["mel_post"][0].cpu()
            dur_pred = outputs["dur_pred"][0].cpu().numpy()
            pitch_pred = outputs["pitch_pred"][0].cpu().numpy()
            energy_pred = outputs["energy_pred"][0].cpu().numpy()

            # 1. Synthesize Audio
            wav = mel_to_audio(mel_post)
            out_wav_path = os.path.join(audio_dir, f"{sid}.wav")
            sf.write(out_wav_path, wav, 16000)
            print(f"  [Audio] {sid}: {out_wav_path} ({len(wav)/16000:.2f}s)")

            # 2. Duration Plot
            plt.figure(figsize=(10, 3.5))
            plt.bar(range(len(dur_pred)), dur_pred, color="teal", alpha=0.85)
            plt.title(f"Duration Predictions - {sid}")
            plt.xlabel("Phoneme Token Index")
            plt.ylabel("Frames (Duration)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dur_dir, f"{sid}_duration.png"), dpi=200)
            plt.close()

            # 3. Pitch (F0) Plot
            plt.figure(figsize=(10, 3.5))
            plt.plot(pitch_pred, color="crimson", lw=2)
            plt.title(f"Pitch (F0) Contour - {sid}")
            plt.xlabel("Frame Index")
            plt.ylabel("Normalized F0")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pitch_dir, f"{sid}_pitch.png"), dpi=200)
            plt.close()

            # 4. Energy Plot
            plt.figure(figsize=(10, 3.5))
            plt.plot(energy_pred, color="darkorange", lw=2)
            plt.title(f"Energy Contour - {sid}")
            plt.xlabel("Frame Index")
            plt.ylabel("Normalized Energy")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(energy_dir, f"{sid}_energy.png"), dpi=200)
            plt.close()

    print("\nINFERENCE & VARIANCE PLOT GENERATION COMPLETE!")

if __name__ == "__main__":
    main()

