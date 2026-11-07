"""
Inference Script for FastSpeech 2 (Telugu)
Synthesizes speech for the 5 assignment test sentences (T1, T2a, T2b, T3, T4).
Generates variance adaptor plots for Duration, Pitch (F0), and Energy.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import VOCAB, text_to_ids
from train import FastSpeech2

TEST_TE = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2_a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2_b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
}

def bar_plot(kind, labels, vals, title, path, ylabel):
    w = max(10, 0.3 * len(labels))
    fig, ax = plt.subplots(figsize=(w, 4))
    x = np.arange(len(labels))
    if kind == "duration":
        ax.bar(x, vals, color="#1f77b4", alpha=0.85)
    else:
        ax.plot(x, vals, "o-", color="#d62728" if kind == "pitch" else "#2ca02c", ms=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    root = base_dir.parent
    cfg_path = base_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    meta_path = root.parent / "FastSpeech2_TTS/Data/Telugu/meta.json"
    if not meta_path.exists():
        meta_path = base_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSpeech2(len(VOCAB), len(meta["spk_names"]), cfg["model"], meta["var_range"]).to(device)

    ckpt_path = base_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = root.parent / "FastSpeech2_TTS/Telugu/checkpoints/best.pt"

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded FastSpeech 2 checkpoint from: {ckpt_path} (step {ck['step']})")

    # Vocoder
    import bigvgan
    voc = bigvgan.BigVGAN._from_pretrained(
        model_id=cfg["vocoder_repo"],
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=os.environ.get("HF_TOKEN", None),
        use_cuda_kernel=False
    )
    voc.remove_weight_norm()
    voc = voc.eval().to(device)

    results_dir = root / "Results" / "Telugu"
    aud_dir = results_dir / "test_generated_audios"
    dur_dir = results_dir / "variance_adaptor_plots" / "duration"
    pit_dir = results_dir / "variance_adaptor_plots" / "pitch"
    ene_dir = results_dir / "variance_adaptor_plots" / "energy"

    for d in [aud_dir, dur_dir, pit_dir, ene_dir]:
        d.mkdir(parents=True, exist_ok=True)

    h = meta["vocoder_mel"]
    sr = h["sampling_rate"]
    frame_ms = h["hop_size"] / sr * 1000

    print("Synthesizing Telugu test sentences...")
    for tid, text in TEST_TE.items():
        nt, toks, ids = text_to_ids(text)
        tok_tensor = torch.from_numpy(ids)[None].to(device)
        spk_tensor = torch.tensor([0], device=device)
        len_tensor = torch.tensor([len(ids)], device=device)

        with torch.no_grad():
            o = model(tok_tensor, spk_tensor, len_tensor)
            mel_denorm = o["mel_post"][0].float() * meta["mel_std"] + meta["mel_mean"]
            wav = voc(mel_denorm.T[None]).squeeze().clamp(-1, 1).cpu().numpy()

        dur = o["d"][0].cpu().numpy()
        pitch = o["p"][0].float().cpu().numpy()
        energy = o["e"][0].float().cpu().numpy()
        st = meta["stats"]["0"]
        hz = np.exp(pitch * st["f0_std"] + st["f0_mean"])

        # Save audio
        sf.write(str(aud_dir / f"{tid}.wav"), wav, sr, subtype="PCM_16")

        # Save plots
        bar_plot("duration", toks, dur * frame_ms, f"Duration - {tid}", dur_dir / f"{tid}.png", "Duration (ms)")
        bar_plot("pitch", toks, hz, f"Pitch (F0) - {tid}", pit_dir / f"{tid}.png", "Pitch F0 (Hz)")
        bar_plot("energy", toks, energy, f"Energy - {tid}", ene_dir / f"{tid}.png", "Energy (z)")
        print(f"  Generated {tid}.wav ({len(wav)/sr:.2f}s)")

    print("Telugu inference complete!")

if __name__ == "__main__":
    main()

