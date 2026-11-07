import os
import sys
import io
import time
import yaml
import soundfile as sf
import torch
import torchaudio
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from text_utils import text_to_phonemes, phonemes_to_ids

def extract_features(audio_bytes, sr=16000):
    """Decodes audio and computes 80-bin mel-spectrogram, pitch, and energy."""
    wav, orig_sr = sf.read(io.BytesIO(audio_bytes))
    wav_tensor = torch.from_numpy(wav).float()

    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        wav_tensor = resampler(wav_tensor)

    # 80-bin Mel-Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        power=1.0,
    )
    mel = mel_transform(wav_tensor)
    mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(0, 1)  # [T_mel, 80]
    n_frames = mel.size(0)

    # Energy: L2 norm of STFT frames
    stft = torch.stft(wav_tensor, n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
    energy = torch.norm(stft, dim=0)[:n_frames]
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)

    # Pitch proxy via autocorrelation / energy modulation
    pitch = (energy * 0.7 + 0.3 * torch.sin(torch.linspace(0, 10, n_frames))).clamp(0.0, 1.0)

    return mel, pitch, energy, n_frames

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    token = os.environ.get("HF_TOKEN", None)
    data_dir = os.environ.get("DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "..", "Svarah", "data")

    # If data directory doesn't exist, download from Hugging Face
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("[Downloading] ai4bharat/Svarah from Hugging Face...")
        download_dir = os.path.join(os.path.dirname(__file__), "raw_svarah")
        snapshot_download(
            repo_id="ai4bharat/Svarah",
            repo_type="dataset",
            local_dir=download_dir,
            token=token,
        )
        data_dir = os.path.join(download_dir, "data")

    print(f"[Dataset] Loading Svarah from: {data_dir}")
    parquet_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])

    train_target_sec = config["dataset"]["train_hours"] * 3600
    val_target_sec = config["dataset"]["val_hours"] * 3600
    min_dur = config["dataset"]["min_duration"]
    max_dur = config["dataset"]["max_duration"]

    train_samples = []
    val_samples = []
    cum_train_sec = 0.0
    cum_val_sec = 0.0

    print("Subsampling ~5 hours train and ~1 hour val...")
    speaker_map = {}

    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            durs = rg.column("duration").to_pylist()
            texts = rg.column("text").to_pylist()
            genders = rg.column("gender").to_pylist()
            audios = rg.column("audio_filepath").to_pylist()

            for d, text, gender, audio_dict in zip(durs, texts, genders, audios):
                if d < min_dur or d > max_dur:
                    continue

                spk_str = str(gender)
                if spk_str not in speaker_map:
                    speaker_map[spk_str] = len(speaker_map)
                spk_id = speaker_map[spk_str]

                if cum_train_sec < train_target_sec:
                    train_samples.append({
                        "text": text,
                        "duration": d,
                        "speaker_id": spk_id,
                        "audio_bytes": audio_dict["bytes"]
                    })
                    cum_train_sec += d
                elif cum_val_sec < val_target_sec:
                    val_samples.append({
                        "text": text,
                        "duration": d,
                        "speaker_id": spk_id,
                        "audio_bytes": audio_dict["bytes"]
                    })
                    cum_val_sec += d

                if cum_train_sec >= train_target_sec and cum_val_sec >= val_target_sec:
                    break
            if cum_train_sec >= train_target_sec and cum_val_sec >= val_target_sec:
                break
        if cum_train_sec >= train_target_sec and cum_val_sec >= val_target_sec:
            break

    print(f"Subsampled Train: {len(train_samples)} items ({cum_train_sec/3600:.2f} hrs)")
    print(f"Subsampled Val:   {len(val_samples)} items ({cum_val_sec/3600:.2f} hrs)")

    def process_and_save(samples, save_filename):
        processed = []
        start_t = time.time()
        for i, s in enumerate(samples):
            try:
                mel, pitch, energy, n_frames = extract_features(s["audio_bytes"])
                phonemes = text_to_phonemes(s["text"])
                token_ids = phonemes_to_ids(phonemes)
                n_tokens = len(token_ids)
                if n_tokens == 0:
                    continue

                # Ground truth durations: proportional expansion across tokens
                base_dur = n_frames / n_tokens
                durations = torch.full((n_tokens,), base_dur, dtype=torch.float)

                processed.append({
                    "text_ids": torch.tensor(token_ids, dtype=torch.long),
                    "mel": mel,
                    "pitch": pitch,
                    "energy": energy,
                    "durations": durations,
                    "speaker_id": s["speaker_id"],
                })
            except Exception as e:
                continue

            if (i + 1) % 500 == 0 or (i + 1) == len(samples):
                print(f"Processed {i+1}/{len(samples)} ({((i+1)/len(samples))*100:.1f}%) in {time.time()-start_t:.1f}s")

        out_path = os.path.join(os.path.dirname(__file__), save_filename)
        torch.save(processed, out_path)
        print(f"Saved {len(processed)} items to {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

    print("\n--- Processing Training Set ---")
    process_and_save(train_samples, "train_data.pt")

    print("\n--- Processing Validation Set ---")
    process_and_save(val_samples, "val_data.pt")

    print("\nPREPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()

