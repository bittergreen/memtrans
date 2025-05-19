from collections import Counter
import torch


class Tokenizer:

    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.stoi = {}  # string to index
        self.itos = {}  # index to string
        
    def fit(self, texts):
        # Count all characters/words
        counter = Counter()
        for text in texts:
            counter.update(text)
        
        # If vocab_size is None, use all unique tokens in the dataset
        vocab_size = self.vocab_size if self.vocab_size is not None else len(counter)
        
        # Create vocabulary from most common characters/words
        vocab = [char for char, _ in counter.most_common(vocab_size)]
        
        # Create mappings
        self.stoi = {char: i for i, char in enumerate(vocab)}
        self.itos = {i: char for i, char in enumerate(vocab)}
        
        # Store the actual vocabulary size
        self.vocab_size = len(vocab)
        
    def encode(self, text):
        # Convert to indices
        indices = [self.stoi.get(char, 0) for char in text]  # Use 0 as default for unknown tokens
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        # Convert indices to text
        text = ''.join([self.itos.get(idx, '') for idx in indices])
        return text

