"""
Hyperparameters for the Memorizing Transformer model.
"""
import torch

# Model Architecture Parameters
n_embed = 16  # size of the embedding vector (C)
n_blocks = 12  # number of transformer blocks
n_heads = 8  # number of attention heads
head_size = 64  # size of each attention head
vocab_size = 200

# Training Parameters
dropout = 0.2  # dropout rate for regularization

# Memory Parameters
mem_size = 1000000  # maximum number of memories to store
topk = 2  # number of memories to retrieve

# Positional Embedding Parameters
scale = head_size ** 0.5  # scale factor for the positional embedding
max_distance = 10000  # maximum distance for the positional embedding
n_buckets = 32  # number of buckets for the positional embedding


# Device Configuration
def get_device():
    """Get the appropriate device for training (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()

