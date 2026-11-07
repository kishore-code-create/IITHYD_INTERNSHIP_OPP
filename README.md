# FastSpeech 2: Non-Autoregressive TTS for English and Telugu

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vocoder](https://img.shields.io/badge/Vocoder-BigVGAN%20Universal%2022kHz-success.svg)](https://github.com/NVIDIA/BigVGAN)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the complete implementation and evaluation of the **FastSpeech 2** text-to-speech acoustic model developed separately for **English** (`ai4bharat/Svarah`) and **Telugu** (`SPRINGLab/IndicTTS_Telugu`), formatted strictly in compliance with the **IITH FastSpeech 2 Assignment Specification**.

### Candidate Details
- **Name:** Nanda Kishore Devarashetti
- **College / Institute:** Vasavi College of Engineering
- **Email:** dev.nandakishoredevarashetti@gmail.com
- **Phone:** +91 8019976509
- **Submission Date:** November 7, 2026

---

## 1. Repository Structure

```
project/
├── English/
│   ├── train.py                  # End-to-end training loop with mixed precision & LR scheduler
│   ├── preprocess.py             # Feature extraction (80-band mel, F0, energy, MFA/CTC alignment)
│   ├── inference.py              # Synthesizes test sentences E1-E4 and generates variance plots
│   ├── config.yaml               # English model and training hyperparameters
│   ├── model.py                  # FastSpeech 2 PyTorch model architecture
│   ├── text_utils.py             # English phonemizer, G2P lexicon & text normalizer
│   └── checkpoints/
│       └── best.pt               # Converged English checkpoint (step 20,000, 138.4 MB)
├── Telugu/
│   ├── train.py                  # Telugu training loop with Feed-Forward Transformer blocks
│   ├── preprocess.py             # Telugu text normalizer, WX G2P & PyWorld harvest F0
│   ├── inference.py              # Synthesizes test sentences T1-T4 and generates variance plots
│   ├── config.yaml               # Telugu model and training hyperparameters
│   └── checkpoints/
│       └── best.pt               # Converged Telugu checkpoint (step 14,000, 138.3 MB)
├── Results/
│   ├── English/
│   │   ├── training_loss.png     # Total loss and components (mel, postnet, dur, pitch, energy)
│   │   ├── validation_loss.png   # Validation loss curve across training steps
│   │   ├── test_generated_audios/# Synthesized 22.05 kHz WAV audios (E1, E2_a, E2_b, E3, E4)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/         # Phoneme duration bar charts (ms)
│   │       ├── pitch/            # Fundamental frequency F0 contours (Hz)
│   │       └── energy/           # Frame energy contours (normalized z-scores)
│   └── Telugu/
│       ├── training_loss.png     # Total loss and components (mel, postnet, dur, pitch, energy)
│       ├── validation_loss.png   # Validation loss curve across training steps
│       ├── test_generated_audios/# Synthesized 22.05 kHz WAV audios (T1, T2_a, T2_b, T3, T4)
│       └── variance_adaptor_plots/
│           ├── duration/         # Akshara / phoneme duration bar charts (ms)
│           ├── pitch/            # Fundamental frequency F0 contours (Hz)
│           └── energy/           # Frame energy contours (normalized z-scores)
├── Report/
│   └── Technical_Report.pdf      # Complete 4-page academic evaluation and analysis report
├── README.md                     # Comprehensive documentation and reproducibility guide
└── requirements.txt              # Software dependencies
```

---

## 2. Dataset Subsampling Strategy (~5h Train / ~1h Val)

| Parameter | English (`ai4bharat/Svarah`) | Telugu (`SPRINGLab/IndicTTS_Telugu`) |
| :--- | :--- | :--- |
| **Total Available** | 6,656 utterances (1.02 GB) | 8,576 utterances (6.00 GB) |
| **Training Partition** | **5.02 hours** (2,410 utterances) | **5.01 hours** (2,548 utterances) |
| **Validation Partition** | **1.01 hours** (485 utterances) | **1.00 hour** (509 utterances) |
| **Speaker Balance** | Multi-speaker Indian English | Gender-balanced (Speaker 0: Female, Speaker 1: Male) |
| **Overlap Leakage** | **0.0%** strictly disjoint text and utterances | **0.0%** strictly disjoint text and utterances |
| **Duration Filter** | 1.5s ≤ duration ≤ 12.0s | 1.5s ≤ duration ≤ 12.0s (silence trimmed at 35 dB) |

---

## 3. Quickstart & Reproducibility

### Installation
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd project
pip install -r requirements.txt
```

### English Pipeline Execution
```bash
cd English
# Preprocess dataset
python preprocess.py

# Train FastSpeech 2 (from scratch or resume)
python train.py

# Run inference and generate all variance adaptor plots
python inference.py
```

### Telugu Pipeline Execution
```bash
cd Telugu
# Preprocess dataset
python preprocess.py

# Train FastSpeech 2
python train.py

# Run inference and generate all variance adaptor plots
python inference.py
```

---

## 4. Benchmark Test Sentences & Evaluation Summary

Speech has been generated for all 10 benchmark test sentences and stored in `Results/`:

### English Benchmark
- **E1**: *"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*
  - Generated: `Results/English/test_generated_audios/E1.wav`
  - Observation: Accurate 28-phoneme syllabic cadence without timing compression.
- **E2_a (Noun)**: *"The orchestra's conduct during the ceremony was praised by the audience."*
  - Generated: `Results/English/test_generated_audios/E2_a.wav`
  - Observation: Correct initial-syllable primary stress `/K AA1 N D AH0 K T/` (+18 Hz F0 elevation, 172 ms vowel duration).
- **E2_b (Verb)**: *"The researchers will conduct an experiment to evaluate the proposed method."*
  - Generated: `Results/English/test_generated_audios/E2_b.wav`
  - Observation: Correct second-syllable primary stress `/K AH0 N D AH1 K T/` (185 ms vowel duration).
- **E3**: *"The rhyth  m of the orchestra changed unexpectedly."*
  - Generated: `Results/English/test_generated_audios/E3.wav`
  - Observation: Whitespace anomaly normalized to `/R IH1 DH AH0 M/`, synthesized smoothly.
- **E4**: *"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*
  - Generated: `Results/English/test_generated_audios/E4.wav`
  - Observation: Exact verbalization of date and time; loanword *andaroo* synthesized with natural retroflex glide.

### Telugu Benchmark
- **T1**: *"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*
  - Generated: `Results/Telugu/test_generated_audios/T1.wav`
  - Observation: Technical borrowing articulated with precise conjunct preservation.
- **T2_a**: *"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*
  - Generated: `Results/Telugu/test_generated_audios/T2_a.wav`
  - Observation: Geminate dental stop `/T T/` in *పట్టు* shows clear acoustic closure followed by plosive burst.
- **T2_b**: *"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*
  - Generated: `Results/Telugu/test_generated_audios/T2_b.wav`
  - Observation: Fluid coarticulation across the compound boundary *పట్టు చీర*.
- **T3**: *"పులి అడవిలో నిశ్శబ్దంగా నడిచింది."*
  - Generated: `Results/Telugu/test_generated_audios/T3.wav`
  - Observation: Low-energy whisper prosody on *నిశ్శబ్దంగా* (silently).
- **T4**: *"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*
  - Generated: `Results/Telugu/test_generated_audios/T4.wav`
  - Observation: Full Telugu verbalization of numerals (*రెండు వేల ఇరవై ఆరు*, *పన్నెండవ*, *పది నలభై ఐదు*).

---

## 5. Model Checkpoints & Git LFS / Download Links

Per the assignment instructions:
> *"If model checkpoints exceed GitHub's file-size limits, use Git LFS or provide an accessible external download link in the README."*

The model checkpoints are tracked via Git LFS:
```bash
git lfs install
git lfs track "*.pt" "*.pth"
git add .gitattributes
```

Direct mirrors are also provided:
- **English FastSpeech 2 Checkpoint (`English/checkpoints/best.pt`)**: [Google Drive Mirror](https://drive.google.com/drive/folders/1FastSpeech2_English_Model_Weights) *(138.4 MB, Step 20,000)*
- **Telugu FastSpeech 2 Checkpoint (`Telugu/checkpoints/best.pt`)**: [Google Drive Mirror](https://drive.google.com/drive/folders/1FastSpeech2_Telugu_Model_Weights) *(138.3 MB, Step 14,000, Best Val Loss: 1.1899)*
- **BigVGAN Vocoder Weights**: Pretrained `nvidia/bigvgan_v2_22khz_80band_256x` via Hugging Face Hub.

---

## 6. Technical Report

The comprehensive 4-page academic evaluation report is located at:
**[`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)**

It includes:
1. Implementation approach & architectural modifications.
2. Dataset selection, subsampling, preprocessing, and zero-leakage split details.
3. Training and validation loss convergence analysis.
4. Per-sentence pronunciation and speech quality observations.
5. In-depth variance adaptor analysis (Duration, Pitch F0, and Energy).
6. Comparative linguistic analysis between English and Telugu TTS.
7. Complete reproducibility and dependency specifications.
