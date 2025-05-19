import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from memtrans.model import Transformer, Tokenizer
import numpy as np
from tqdm import tqdm
import os
import argparse

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_EPOCHS = 3
EVAL_INTERVAL = 100
SAVE_INTERVAL = 1000

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, window_size):
        self.texts = texts
        self.tokenizer = tokenizer
        self.window_size = window_size
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Encode the text
        tokens = self.tokenizer.encode(text)
        
        # Create input and target sequences
        x = tokens[:-1]
        y = tokens[1:]
        
        # Pad sequences if needed
        if len(x) < self.window_size:
            pad_len = self.window_size - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])
        else:
            x = x[:self.window_size]
            y = y[:self.window_size]
            
        return x, y

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            _, loss = model(x, y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load and preprocess data
    train_texts = load_data(args.train_file)
    val_texts = load_data(args.val_file)
    
    # Initialize tokenizer and fit on training data
    tokenizer = Tokenizer()
    tokenizer.fit(train_texts)
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, window_size=8)
    val_dataset = TextDataset(val_texts, tokenizer, window_size=8)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = Transformer(n_heads=4, head_size=8, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokenizer_stoi': tokenizer.stoi,
                'tokenizer_itos': tokenizer.itos,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            
        # Save checkpoint periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokenizer_stoi': tokenizer.stoi,
                'tokenizer_itos': tokenizer.itos,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

if __name__ == '__main__':
    main() 