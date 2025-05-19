import requests
import os
import random
from pathlib import Path
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL verification warnings
urllib3.disable_warnings(InsecureRequestWarning)

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


def download_text(url, output_path):
    """Download text from a URL and save it to a file."""
    try:
        # Try with SSL verification first
        response = requests.get(url, verify=True)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        # If SSL verification fails, try without verification
        print(f"SSL verification failed for {url}, trying without verification...")
        response = requests.get(url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    return output_path

def split_data(input_file, train_ratio=0.8):
    """Split the data into training and validation sets."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and filter lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Shuffle the lines
    random.shuffle(lines)
    
    # Split into train and validation
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Write to separate files
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / 'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(data_dir / 'val.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    return str(data_dir / 'train.txt'), str(data_dir / 'val.txt')

def main():
    # URLs for some public domain texts from Project Gutenberg
    urls = [
        'https://www.gutenberg.org/files/1342/1342-0.txt',  # Pride and Prejudice
        'https://www.gutenberg.org/files/11/11-0.txt',      # Alice's Adventures in Wonderland
        'https://www.gutenberg.org/files/84/84-0.txt',      # Frankenstein
    ]
    
    # Download and prepare each text
    successful_downloads = []
    for i, url in enumerate(urls):
        print(f"Downloading text {i+1}/{len(urls)}...")
        output_path = f'data/raw_text_{i}.txt'
        if download_text(url, output_path):
            successful_downloads.append(i)
    
    if not successful_downloads:
        print("No files were successfully downloaded. Please check your internet connection and try again.")
        return
    
    # Combine all texts
    combined_text = []
    for i in successful_downloads:
        try:
            with open(f'data/raw_text_{i}.txt', 'r', encoding='utf-8') as f:
                combined_text.extend(f.readlines())
        except Exception as e:
            print(f"Error reading file data/raw_text_{i}.txt: {e}")
    
    if not combined_text:
        print("No text data was successfully read. Please check the downloaded files.")
        return
    
    # Write combined text
    with open('data/combined.txt', 'w', encoding='utf-8') as f:
        f.writelines(combined_text)
    
    # Split into train and validation sets
    train_path, val_path = split_data('data/combined.txt')
    
    print(f"\nDataset prepared successfully!")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"\nTo train the model, run:")
    print(f"python train.py --train_file {train_path} --val_file {val_path}")

if __name__ == '__main__':
    main() 