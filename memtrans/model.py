from memory import Memory
from tokenizer import Tokenizer
from positional_embedding import T5PositionalEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from hyperparameters import (
    n_embed, dropout, n_blocks,
    n_heads, head_size, vocab_size, device
)


"""
First we need to get things clear. I mean the structure of the network or sth.

Input
Say we have a token sequence to go in, we chunk it to head_size tokens, each token has the size n_embed.
So the input shall be (B, T, C), T is time - head_size, C is channel - n_embed

Attention Head
In each head, we're gonna have q, k, v.
We use q @ k to find the attention weights, and then use the masked and normalized attention weights to attend to the v.
The output of each head is (B, T, head_size), and we have n_heads of them.

MultiHeadAttention
We concat the outputs of the n_heads, and then use a linear layer to project the concatenated output to the n_embed.

FeedForward
We use a linear layer to project the input to the n_embed, and then use a ReLU activation function to activate the output.
"""


class MultiHeadAttention(nn.Module):

    """
    Attention heads with KNN memory available and Transformer-XL recurrence mechanism
    """
    def __init__(self, n_heads=n_heads, head_size=head_size, memorizing=False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.query = nn.Linear(n_embed, n_heads * head_size, device=device)
        self.key = nn.Linear(n_embed, n_heads * head_size, device=device)
        self.value = nn.Linear(n_embed, n_heads * head_size, device=device)
        self.output = nn.Linear(n_heads * head_size, n_embed, device=device)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_size ** 0.5
        
        self.memorizing = memorizing
        if self.memorizing:
            self.memory = Memory()
            # Learnable parameter for blending between basic attention and memory attention
            self.gate_bias = nn.Parameter(torch.randn((self.n_heads, 1, 1), device=device))

    def forward(self, x, xl_mem=None, rel_pos=None):  # x -> (B, T, C), xl_mem -> (B, M, 2, n_heads * head_size)
        # Basic masked attention
        q = self.query(x)  # (B, T, n_heads * head_size)
        k = self.key(x)    # (B, T, n_heads * head_size)
        v = self.value(x)  # (B, T, n_heads * head_size)

        # Add normalization for better retrieval from the vector db
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Store current k and v
        current_k = k
        current_v = v
        
        # If we have memories from previous segments, concatenate them(Transformer-XL recurrence)
        if xl_mem is not None:
            xl_key, xl_value = xl_mem.unbind(dim=-1)  # (B, M, n_heads * head_size), (B, M, n_heads * head_size)
            # Concatenate current and memory keys/values(XL)
            k = torch.cat([xl_key, k], dim=1)  # (B, M+T, n_heads * head_size)
            v = torch.cat([xl_value, v], dim=1)  # (B, M+T, n_heads * head_size)

        qk = q @ k.transpose(-2, -1) * self.scale  # (B, T, M+T)

        # Add in relative positions
        i, j = qk.shape[-2:]
        if rel_pos is not None:
            qk = rel_pos[..., -i:, -j:] + qk

        # Create attention mask for the extended sequence
        mask = torch.ones(i, j, dtype=torch.bool, device=device).triu(diagonal=j-i+1)
        qk = qk.masked_fill(mask, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qk = self.dropout(qk)
        out = qk @ v  # (B, T, head_size)

        # For the memorizing layer heads, include attention to the KNN memory
        if self.memorizing:
            if not self.memory.is_empty():
                # retrieved kv are of size (B, T, topk, n_heads * head_size)
                retrieved_k, retrieved_v = self.memory.retrieve(q)
                # (B, T, n_heads * head_size) @ (B, T, n_heads * head_size, topk) -> (B, T, topk)
                wei = einsum(q, retrieved_k, "b t c, b t c k -> b t k") * self.scale
                wei = F.softmax(wei, dim=-1)
                wei = self.dropout(wei)
                mem_att = einsum(wei, retrieved_v, "b t k, b t c k -> b t c")

                # combine the two attention results
                mem_att = rearrange(mem_att, "b t (h d) -> b h t d", h=n_heads, d=head_size)
                out = rearrange(out, "b t (h d) -> b h t d", h=n_heads, d=head_size)
                gate = torch.sigmoid(self.gate_bias)
                out = mem_att * gate + out * (1 - gate)
                out = rearrange(out, 'b h t d -> b t (h d)')
            # store current kv anyways
            self.memory.store(current_k, current_v)

        # XL recurrence
        xl_recurrence = torch.stack([current_k, current_v], dim=-1)  # (B, T, 2, n_heads * head_size)
        out = self.output(out)

        return out, xl_recurrence


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, device=device),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed, device=device),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_heads, head_size, memorizing=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed, device=device)
        self.mha = MultiHeadAttention(n_heads, head_size, memorizing)
        self.ln2 = nn.LayerNorm(n_embed, device=device)
        self.ff = FeedForward(n_embed)

    def forward(self, x, xl_mem=None, rel_pos=None):
        # w/ residual connection
        residual = x
        x, new_xl_recurrence = self.mha(self.ln1(x), xl_mem, rel_pos)
        x += residual
        x = x + self.ff(self.ln2(x))
        return x, new_xl_recurrence


class MemorizingTransformer(nn.Module):

    """
    x = tokenizer(x)
    x = embedding(x)
    x += positional_embedding
    """

    def __init__(
            self,
            n_embed=n_embed,
            n_heads=n_heads,
            head_size=head_size,
            n_blocks=n_blocks,
            vocab_size=vocab_size
    ):

        super().__init__()
        self.tokenizer = Tokenizer(vocab_size)
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        
        # Initialize biologically-inspired positional encoding
        self.pos_emb = T5PositionalEmbedding()
        self.mem_pos_emb = T5PositionalEmbedding()

        self.blocks = nn.ModuleList(
            [Block(n_heads, head_size, True if i == (n_blocks - 2) else False) for i in range(n_blocks)])
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, x, rel_pos=None, xl_mem=None, targets=None):

        x = self.tokenizer.encode(x)

        batch_size, sequence_length = x.shape[0], x.shape[1]
        rel_pos = self.pos_emb(sequence_length)

        for block in self.blocks:
            x, xl_mem = block(x, xl_mem, rel_pos)


    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last window_size tokens
            idx_cond = idx[:, -window_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == '__main__':
    block = Block(n_heads, head_size, True)
    b = 8
    t = head_size
    c = n_embed
    x = torch.randn((b, t, c), dtype=torch.float32, device=device)
    out1, xl_rec1 = block(x)
    out2, xl_rec2 = block(out1, xl_mem=xl_rec1)
    print(out2.shape)

