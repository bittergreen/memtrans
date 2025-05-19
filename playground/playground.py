import torch
from torch.utils.data import Dataset
from typing import List, Union
import os
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts: Union[List[str], str], tokenizer=None, max_length: int = 512):
        """
        Initialize the dataset with text data.
        
        Args:
            texts: Either a list of strings or a path to a text file
            tokenizer: Optional tokenizer to process the text
            max_length: Maximum sequence length for tokenization
        """
        if isinstance(texts, str) and os.path.isfile(texts):
            # Load from file
            with open(texts, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
        else:
            # Assume it's a list of strings
            self.texts = texts
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer is not None:
            # Tokenize the text if a tokenizer is provided
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        else:
            # Return raw text if no tokenizer
            return {'text': text}

# Example usage:
if __name__ == "__main__":
    # Example 1: Using list of texts with BERT tokenizer
    texts = ["This is the first text.", "This is the second text."]
    
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset with tokenizer
    dataset = TextDataset(texts, tokenizer=tokenizer)
    
    # Print some information about the tokenized data
    print(f"Dataset length: {len(dataset)}")
    first_item = dataset[0]
    print(f"First item input_ids shape: {first_item['input_ids'].shape}")
    print(f"First item attention_mask shape: {first_item['attention_mask'].shape}")
    
    # Example 2: Using different tokenizers
    # GPT-2 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # RoBERTa tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Example 3: Using with DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a batch of data
    batch = next(iter(dataloader))
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
