import torch
import torch.nn as nn
import math

from hyperparameters import scale, max_distance, n_buckets, n_heads


class T5PositionalEmbedding(nn.Module):

    def __init__(self, scale=scale, max_distance=max_distance, n_buckets=n_buckets, n_heads=n_heads):
        super().__init__()

        self.scale = scale
        self.max_distance = max_distance
        self.n_buckets = n_buckets
        self.n_heads = n_heads

        self.relative_position_bucket_embedding = nn.Embedding(n_buckets, n_heads)
    
    def _relative_position_bucket(self, relative_position):
        max_exact = self.n_buckets // 2
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))
        
        is_exact = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)  # log of matrix divided by scalar 
            / math.log(self.max_distance / max_exact) * (self.n_buckets - max_exact) # scalar
            ).long() # convert float to int
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1))

        return torch.where(is_exact, n, val_if_large)
    
    def forward(self, sequence_length):
        q_pos = torch.arange(sequence_length, dtype=torch.long)
        q_pos = q_pos.reshape(q_pos.shape[0], 1)
        # 2 times sequence length for Transformer-XL context
        k_pos = torch.arange(2 * sequence_length, dtype=torch.long)
        rel_pos = k_pos - q_pos
        pos_indices = self._relative_position_bucket(rel_pos)
        pos_embedding_values = self.relative_position_bucket_embedding(pos_indices)  # (sequence, context, heads)
        # convert to (batch, heads, sequence, context)
        pos_embedding_values = pos_embedding_values.transpose(0, 2).unsqueeze(0)
        return pos_embedding_values * self.scale

