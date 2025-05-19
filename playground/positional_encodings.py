import torch
import torch.nn as nn
import math
import numpy as np

class DynamicPositionalEncoding(nn.Module):
    """
    Dynamic positional encoding inspired by hippocampal time cells.
    The encoding adapts based on the content and position in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Base positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Dynamic scaling factors
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Get base positional encoding
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        
        # Compute dynamic scaling based on content
        scale = self.scale_net(x)  # (batch_size, seq_len, d_model)
        scale = torch.sigmoid(scale)  # Normalize to [0, 1]
        
        # Apply dynamic scaling
        pos_enc = pos_enc * (1 + scale)  # Scale the positional encoding
        
        return pos_enc

class PhaseBasedEncoding(nn.Module):
    """
    Phase-based encoding inspired by hippocampal theta oscillations and phase precession.
    """
    def __init__(self, d_model, max_len=5000, n_oscillations=4):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_oscillations = n_oscillations
        
        # Initialize oscillation frequencies
        self.frequencies = nn.Parameter(torch.randn(n_oscillations) * 0.1 + 1.0)
        self.phases = nn.Parameter(torch.randn(n_oscillations) * 0.1)
        
        # Projection to d_model
        self.proj = nn.Linear(n_oscillations * 2, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Create time steps
        t = torch.arange(seq_len, device=x.device).float()
        
        # Generate oscillations
        oscillations = []
        for i in range(self.n_oscillations):
            phase = self.phases[i]
            freq = self.frequencies[i]
            sin_wave = torch.sin(2 * math.pi * freq * t + phase)
            cos_wave = torch.cos(2 * math.pi * freq * t + phase)
            oscillations.extend([sin_wave, cos_wave])
        
        # Stack oscillations
        oscillations = torch.stack(oscillations, dim=1)  # (seq_len, n_oscillations*2)
        
        # Project to d_model
        pos_enc = self.proj(oscillations)  # (seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0)  # (1, seq_len, d_model)
        
        return pos_enc

class MultiScaleEncoding(nn.Module):
    """
    Multi-scale encoding inspired by different temporal scales in hippocampus.
    Combines multiple time scales of positional information.
    """
    def __init__(self, d_model, max_len=5000, scales=[1, 4, 16, 64]):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scales = scales
        
        # Create encoding for each scale
        self.scale_encodings = nn.ModuleList([
            nn.Linear(1, d_model // len(scales)) for _ in scales
        ])
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Generate position indices for each scale
        pos_encodings = []
        for i, scale in enumerate(self.scales):
            # Create positions at this scale
            positions = torch.arange(0, seq_len, scale, device=x.device).float()
            # Interpolate to full sequence length
            positions = positions.unsqueeze(0).repeat(batch_size, 1)
            # Encode positions
            enc = self.scale_encodings[i](positions.unsqueeze(-1))
            pos_encodings.append(enc)
        
        # Combine encodings from all scales
        pos_enc = torch.cat(pos_encodings, dim=-1)  # (batch_size, seq_len, d_model)
        
        return pos_enc

class BiologicallyInspiredEncoding(nn.Module):
    """
    Combined biologically-inspired encoding that uses all three approaches.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Initialize all three encodings
        self.dynamic = DynamicPositionalEncoding(d_model // 3, max_len)
        self.phase = PhaseBasedEncoding(d_model // 3, max_len)
        self.multiscale = MultiScaleEncoding(d_model // 3, max_len)
        
    def forward(self, x):
        # Get encodings from all three methods
        dynamic_enc = self.dynamic(x)
        phase_enc = self.phase(x)
        multiscale_enc = self.multiscale(x)
        
        # Combine all encodings
        pos_enc = torch.cat([dynamic_enc, phase_enc, multiscale_enc], dim=-1)
        
        return pos_enc 