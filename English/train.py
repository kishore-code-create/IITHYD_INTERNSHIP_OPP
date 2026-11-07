import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import FastSpeech2
from text_utils import ALL_SYMBOLS

class TTSDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    text_lens = [b["text_ids"].size(0) for b in batch]
    mel_lens = [b["mel"].size(0) for b in batch]

    max_text_len = max(text_lens)
    max_mel_len = max(mel_lens)

    padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    padded_mel = torch.zeros(len(batch), max_mel_len, 80, dtype=torch.float)
    padded_pitch = torch.zeros(len(batch), max_mel_len, dtype=torch.float)
    padded_energy = torch.zeros(len(batch), max_mel_len, dtype=torch.float)
    padded_dur = torch.zeros(len(batch), max_text_len, dtype=torch.float)
    speakers = torch.tensor([b["speaker_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        t_len = b["text_ids"].size(0)
        m_len = b["mel"].size(0)

        padded_text[i, :t_len] = b["text_ids"]
        padded_mel[i, :m_len] = b["mel"]
        padded_pitch[i, :min(m_len, len(b["pitch"]))] = b["pitch"][:m_len]
        padded_energy[i, :min(m_len, len(b["energy"]))] = b["energy"][:m_len]
        padded_dur[i, :t_len] = b["durations"]

    return {
        "text": padded_text,
        "mel": padded_mel,
        "pitch": padded_pitch,
        "energy": padded_energy,
        "duration": padded_dur,
        "speaker": speakers,
    }

def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_pt = os.path.join(base_dir, "train_data.pt")
    val_pt = os.path.join(base_dir, "val_data.pt")

    if not os.path.exists(train_pt) or not os.path.exists(val_pt):
        print("Data files not found. Running preprocessing first...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(base_dir, "preprocess.py")], check=True)

    train_loader = DataLoader(TTSDataset(train_pt), batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TTSDataset(val_pt), batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = FastSpeech2(vocab_size=len(ALL_SYMBOLS), n_speakers=config["model"]["speaker_embedding"]["n_speakers"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and config["train"]["mixed_precision"]))

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    epochs = config["train"]["epochs"]
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, "..", "Results", "English")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Starting FastSpeech 2 training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            text = batch["text"].to(device)
            mel_target = batch["mel"].to(device)
            pitch_target = batch["pitch"].to(device)
            energy_target = batch["energy"].to(device)
            dur_target = batch["duration"].to(device)
            speaker = batch["speaker"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and config["train"]["mixed_precision"])):
                outputs = model(
                    text=text,
                    speaker_id=speaker,
                    dur_target=dur_target,
                    pitch_target=pitch_target,
                    energy_target=energy_target
                )

                # Match lengths for loss computation
                min_t = min(outputs["mel_init"].size(1), mel_target.size(1))
                mel_init_loss = l1_loss(outputs["mel_init"][:, :min_t], mel_target[:, :min_t])
                mel_post_loss = l1_loss(outputs["mel_post"][:, :min_t], mel_target[:, :min_t])

                min_pitch = min(outputs["pitch_pred"].size(1), pitch_target.size(1))
                pitch_loss = mse_loss(outputs["pitch_pred"][:, :min_pitch], pitch_target[:, :min_pitch])

                min_energy = min(outputs["energy_pred"].size(1), energy_target.size(1))
                energy_loss = mse_loss(outputs["energy_pred"][:, :min_energy], energy_target[:, :min_energy])

                log_dur_target = torch.log(dur_target + 1.0)
                dur_loss = mse_loss(outputs["log_dur_pred"], log_dur_target)

                loss = (
                    config["train"]["loss_weights"]["mel_init"] * mel_init_loss +
                    config["train"]["loss_weights"]["mel_post"] * mel_post_loss +
                    config["train"]["loss_weights"]["pitch"] * pitch_loss +
                    config["train"]["loss_weights"]["energy"] * energy_loss +
                    config["train"]["loss_weights"]["duration"] * dur_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_thresh"])
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                text = batch["text"].to(device)
                mel_target = batch["mel"].to(device)
                dur_target = batch["duration"].to(device)
                pitch_target = batch["pitch"].to(device)
                energy_target = batch["energy"].to(device)
                speaker = batch["speaker"].to(device)

                outputs = model(text=text, speaker_id=speaker, dur_target=dur_target)
                min_t = min(outputs["mel_post"].size(1), mel_target.size(1))
                val_loss = l1_loss(outputs["mel_post"][:, :min_t], mel_target[:, :min_t])
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss", color="royalblue", lw=2)
    plt.title("English FastSpeech 2 - Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(val_losses, label="Validation Loss", color="crimson", lw=2)
    plt.title("English FastSpeech 2 - Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "validation_loss.png"), dpi=200)
    plt.close()

    print("\nTRAINING COMPLETE! Best checkpoint and loss plots saved.")

if __name__ == "__main__":
    main()

