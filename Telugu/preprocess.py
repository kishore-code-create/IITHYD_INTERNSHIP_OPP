"""
Preprocessing Pipeline for FastSpeech 2 (Telugu)
Reference Dataset: SPRINGLab/IndicTTS_Telugu
Subsampling: 5.0 hours train, 1.0 hour validation, speaker-balanced, no text overlap
Feature Extraction: 22.05 kHz audio resample, silence trimming, 80-band log mel-spectrogram,
                   continuous F0 via PyWorld (Harvest/DIO), L2 frame energy, and CTC learned alignment.
"""

import os
import sys
import io
import re
import json
import yaml
import unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyworld as pw
except ImportError:
    pw = None

try:
    import librosa
except ImportError:
    librosa = None

# Telugu Phonetics Vocabulary
VOWELS = {"అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"], "ఋ": ["r", "u"],
          "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"]}
MATRAS = {"ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"], "ృ": ["r", "u"],
          "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"]}
CONS = {"క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng", "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh",
        "ఞ": "ny", "ట": "T", "ఠ": "Th", "డ": "D", "ఢ": "Dh", "ణ": "N", "త": "t", "థ": "th", "ద": "d",
        "ధ": "dh", "న": "n", "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m", "య": "y", "ర": "r",
        "ఱ": "R", "ల": "l", "వ": "v", "శ": "sh", "ష": "S", "స": "s", "హ": "h", "ళ": "L"}
VIRAMA = "్"
NASAL, VISARGA = {"ం", "ఁ"}, "ః"
PAUSE_CH = set(",;:.?!")
PAD, SIL, SP = "<pad>", "sil", "sp"
VOCAB = [PAD] + sorted({t for v in VOWELS.values() for t in v} | {t for v in MATRAS.values() for t in v}
                       | set(CONS.values()) | {"m", "h", "a"}) + [SIL, SP]
VOCAB = list(dict.fromkeys(VOCAB))
TOK2ID = {t: i for i, t in enumerate(VOCAB)}

ONES = ["సున్న", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
        "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది",
        "పంతొమ్మిది"]
TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
TE_DIGITS = {ord(c): str(i) for i, c in enumerate("౦౧౨౩౪౫౬౭౮౯")}

def num_te(n):
    if n < 20: return ONES[n]
    if n < 100:
        t, o = divmod(n, 10); return TENS[t] + (" " + ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0: return "వంద" if h == 1 else ONES[h] + " వందలు"
        return ("నూట" if h == 1 else ONES[h] + " వందల") + " " + num_te(r)
    if n < 100000:
        t, r = divmod(n, 1000)
        if r == 0: return "వెయ్యి" if t == 1 else num_te(t) + " వేలు"
        return ("వెయ్యి" if t == 1 else num_te(t) + " వేల") + " " + num_te(r)
    return " ".join(ONES[int(c)] for c in str(n))

def ordinal_te(n):
    w = num_te(n)
    if w.endswith("ై"): return w + "యవ"
    if w[-1] in "ుి": return w[:-1] + "వ"
    return w + "వ"

def normalize_text(t):
    t = unicodedata.normalize("NFC", t).replace("\u200c", "").replace("\u200d", "").translate(TE_DIGITS)
    t = re.sub(r"(\d{1,2}):(\d{2})",
               lambda m: " " + num_te(int(m[1])) + (" " + num_te(int(m[2])) if int(m[2]) else "") + " ", t)
    t = re.sub(r"(\d+)వ(?![\u0C00-\u0C7F])", lambda m: " " + ordinal_te(int(m[1])) + " ", t)
    t = re.sub(r"\d+", lambda m: " " + num_te(int(m[0])) + " ", t)
    t = "".join(ch if ("\u0C00" <= ch <= "\u0C7F") or ch in PAUSE_CH else " " if not ch.isalpha() else ch
                for ch in t)
    return re.sub(r"\s+", " ", t).strip()

def g2p(ntext):
    ch, out, i = list(ntext), [], 0
    while i < len(ch):
        c = ch[i]
        if c in CONS:
            out.append(CONS[c]); nx = ch[i + 1] if i + 1 < len(ch) else ""
            if nx == VIRAMA: i += 2
            elif nx in MATRAS: out += MATRAS[nx]; i += 2
            else: out.append("a"); i += 1
        elif c in VOWELS: out += VOWELS[c]; i += 1
        elif c in NASAL: out.append("m"); i += 1
        elif c == VISARGA: out.append("h"); i += 1
        elif c in PAUSE_CH: out.append(SP); i += 1
        else: i += 1
    res = []
    for t in out:
        if t == SP and res and res[-1] == SP: continue
        res.append(t)
    while res and res[0] == SP: res.pop(0)
    while res and res[-1] == SP: res.pop()
    return [SIL] + res + [SIL]

def text_to_ids(text):
    nt = normalize_text(text); toks = g2p(nt)
    return nt, toks, np.array([TOK2ID[t] for t in toks], dtype=np.int64)

class MelExtractor:
    def __init__(self, sr=22050, n_fft=1024, hop=256, win=1024, n_mel=80):
        self.sr, self.hop, self.n_fft = sr, hop, n_fft
        import librosa
        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel, fmin=0, fmax=None)
        self.fb = torch.from_numpy(fb).float()
        self.win = torch.hann_window(win)

    def __call__(self, y):
        yt = torch.from_numpy(y).float()
        stft = torch.stft(yt, self.n_fft, self.hop, win_length=self.win.size(0),
                          window=self.win, return_complex=True)
        mag = torch.clamp(stft.abs(), min=1e-5)
        mel = torch.log(torch.matmul(self.fb, mag)).T.numpy()
        loge = torch.log(torch.clamp(torch.norm(mag, dim=0), min=1e-5)).numpy()
        return mel, loge

def main():
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print("FastSpeech 2 Telugu Preprocessor Initialized.")
    print("Vocabulary size:", len(VOCAB), "Phoneme tokens:", VOCAB)

if __name__ == "__main__":
    main()

