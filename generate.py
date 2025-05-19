import torch
from memtrans.model import Transformer, Tokenizer
import argparse
import json

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(checkpoint_path, vocab_size):
    model = Transformer(n_heads=4, head_size=8, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_tokenizer(checkpoint_path):
    # Load tokenizer state from checkpoint
    checkpoint = torch.load(checkpoint_path)
    tokenizer = Tokenizer()
    tokenizer.stoi = checkpoint.get('tokenizer_stoi', {})
    tokenizer.itos = checkpoint.get('tokenizer_itos', {})
    tokenizer.vocab_size = len(tokenizer.stoi)
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained transformer model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size used during training')
    parser.add_argument('--prompt', type=str, default='', help='Starting prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = load_model(args.checkpoint, args.vocab_size)
    model = model.to(device)
    model.eval()
    
    tokenizer = load_tokenizer(args.checkpoint)
    
    # Encode prompt
    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
        prompt_tokens = prompt_tokens.unsqueeze(0).to(device)  # Add batch dimension
    else:
        # Start with a single <sos> token if no prompt
        prompt_tokens = torch.tensor([[tokenizer.stoi['<sos>']]], dtype=torch.long).to(device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(prompt_tokens, max_new_tokens=args.max_tokens)
    
    # Decode and print generated text
    generated_text = tokenizer.decode(generated[0].tolist())
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == '__main__':
    main() 