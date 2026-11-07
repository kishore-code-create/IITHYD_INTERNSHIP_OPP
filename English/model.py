import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FFTBlock(nn.Module):
    """Feed-Forward Transformer Block (Multi-Head Attention + 1D Conv FFN)."""
    def __init__(self, d_model, n_head, ffn_dim, kernel_size=9, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, ffn_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(ffn_dim, d_model, kernel_size, padding=padding)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 1D Conv FFN
        conv_in = x.transpose(1, 2)
        conv_out = F.relu(self.conv1(conv_in))
        conv_out = self.conv2(conv_out).transpose(1, 2)
        x = self.norm2(x + self.dropout(conv_out))
        return x

class VariancePredictor(nn.Module):
    def __init__(self, d_model=256, filter_size=256, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        h = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(F.relu(self.norm1(h)))
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(F.relu(self.norm2(h)))
        out = self.linear(h).squeeze(-1)  # [B, T]
        return out

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, duration):
        # x: [B, T, D], duration: [B, T]
        output = []
        for i in range(x.size(0)):
            rep_list = []
            for j in range(x.size(1)):
                d = max(0, int(round(duration[i, j].item())))
                if d > 0:
                    rep_list.append(x[i, j:j+1].repeat(d, 1))
            if rep_list:
                output.append(torch.cat(rep_list, dim=0))
            else:
                output.append(torch.zeros(1, x.size(2), device=x.device))

        # Pad to max length
        max_len = max(o.size(0) for o in output)
        padded = torch.zeros(len(output), max_len, x.size(2), device=x.device)
        for i, o in enumerate(output):
            padded[i, :o.size(0)] = o
        return padded

class PostNet(nn.Module):
    def __init__(self, n_mels=80, channels=256, kernel_size=5, n_convs=5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = []
        layers.append(nn.Conv1d(n_mels, channels, kernel_size, padding=padding))
        layers.append(nn.GroupNorm(1, channels))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(0.2))

        for _ in range(n_convs - 2):
            layers.append(nn.Conv1d(channels, channels, kernel_size, padding=padding))
            layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Conv1d(channels, n_mels, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, mel):
        # mel: [B, T, n_mels]
        residual = self.net(mel.transpose(1, 2)).transpose(1, 2)
        return mel + residual

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size=150, d_model=256, n_head=2, ffn_dim=1024, n_mels=80, n_speakers=128):
        super().__init__()
        # Text Encoder
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_blocks = nn.ModuleList([
            FFTBlock(d_model, n_head, ffn_dim) for _ in range(4)
        ])

        # Multi-speaker Embedding
        self.speaker_emb = nn.Embedding(n_speakers, d_model) if n_speakers > 1 else None

        # Variance Adaptor
        self.duration_predictor = VariancePredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model)
        self.pitch_emb = nn.Embedding(256, d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.energy_emb = nn.Embedding(256, d_model)

        # Mel Decoder
        self.decoder_pos_enc = PositionalEncoding(d_model)
        self.decoder_blocks = nn.ModuleList([
            FFTBlock(d_model, n_head, ffn_dim) for _ in range(4)
        ])
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.postnet = PostNet(n_mels=n_mels)

    def forward(self, text, speaker_id=None, dur_target=None, pitch_target=None, energy_target=None, d_control=1.0, p_control=1.0, e_control=1.0):
        # Text Encoder
        x = self.pos_enc(self.src_emb(text))
        for block in self.encoder_blocks:
            x = block(x)

        if self.speaker_emb is not None and speaker_id is not None:
            spk = self.speaker_emb(speaker_id).unsqueeze(1)
            x = x + spk

        # Duration Predictor
        log_dur_pred = self.duration_predictor(x)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1.0, min=0.0) * d_control

        if dur_target is not None:
            x_reg = self.length_regulator(x, dur_target)
        else:
            x_reg = self.length_regulator(x, dur_pred)

        # Pitch & Energy Predictors
        pitch_pred = self.pitch_predictor(x_reg) * p_control
        energy_pred = self.energy_predictor(x_reg) * e_control

        if pitch_target is not None:
            p_bin = torch.clamp((pitch_target * 255).long(), 0, 255)
        else:
            p_bin = torch.clamp((pitch_pred * 255).long(), 0, 255)

        if energy_target is not None:
            e_bin = torch.clamp((energy_target * 255).long(), 0, 255)
        else:
            e_bin = torch.clamp((energy_pred * 255).long(), 0, 255)

        x_reg = x_reg + self.pitch_emb(p_bin) + self.energy_emb(e_bin)

        # Mel Decoder
        y = self.decoder_pos_enc(x_reg)
        for block in self.decoder_blocks:
            y = block(y)

        mel_init = self.mel_linear(y)
        mel_post = self.postnet(mel_init)

        return {
            "mel_init": mel_init,
            "mel_post": mel_post,
            "log_dur_pred": log_dur_pred,
            "dur_pred": dur_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
        }
