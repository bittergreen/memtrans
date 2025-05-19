import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets


os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"

dataset = datasets.load_dataset("ccdv/arxiv-summarization", split='train', streaming=True)
raw_dataset = list(dataset.take(3500))

segments = 10
segment_length = 512
chunk_size = segments * segment_length
max_iters = 300


raw_dataset = [x['article'] for x in raw_dataset]
# filtering out articles that's shorter than chunk_size
raw_dataset = [x for x in raw_dataset if len(x) > chunk_size]

# tokenizer
all_text = "".join(raw_dataset)
chars = sorted(list(set(all_text)))

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
del all_text

encoded = [np.fromstring(doc, dtype='uint8') for doc in raw_dataset]

all_encoded = np.concatenate(encoded)
c_chars = set(all_encoded)

def clip_article(doc, chunk_size):
    remainder = len(doc) % chunk_size
    return doc[:-remainder]

clipped = [clip_article(doc, chunk_size) for doc in encoded]

clipped = [doc.reshape(-1, chunk_size) for doc in clipped]

processed_data = torch.tensor(np.concatenate(clipped), dtype=torch.long)
processed_data.shape

data_length = processed_data.shape[0]
eighty_split = int(data_length * 0.8)
ninety_split = int(data_length * 0.9)

train_loader = iter(DataLoader(processed_data[:eighty_split], batch_size=8, shuffle=True))
test_loader = iter(DataLoader(processed_data[eighty_split:ninety_split], batch_size=8, shuffle=True))
val_loader = iter(DataLoader(processed_data[ninety_split:], batch_size=8, shuffle=True))

example = next(val_loader)

seq, labels = example[:, :-1], example[:, 1:]

model = nn.Sequential(
    nn.Embedding(70, 16), # vocab_size, embedding_dim
    nn.Linear(16, 150),
    nn.ReLU(),
    nn.Linear(150, 128),
    nn.ReLU()
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
model.train()  # changing the model to training mode


for i in range(max_iters):
    data = next(train_loader)  # (batch_size, sequence_length)  # (8, 5120)
    seq, labels = data[:, :-1], data[:, 1:]
    train_loss = 0.0
    model.train()
    
    # go 10 passes of (8, 512)
    for seq_segment, label_segment in zip(seq.chunk(segments, dim=-1), labels.chunk(segments, dim=-1)):
        optimizer.zero_grad()
        y_logits = model(seq_segment)  # (B, T, C)  # (8, 512, 128)
        y_logits = y_logits.transpose(-2, -1)
        loss = loss_fn(y_logits, label_segment)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if i % 5 == 0:
        print (train_loss / segments)

    if i > 0 and i % 50 == 0:
        val_data = next(val_loader)
        seq, labels = val_data[:, :-1], val_data[:, 1:]
        eval_loss = 0.
        model.eval()
        for seq_segment, labels_segment in zip(seq.chunk(segments, dim = -1), labels.chunk(segments, dim = -1)): # ten passes of (8, 512)
            with torch.no_grad():
                y_pred = model(seq_segment)
                y_pred = y_pred.transpose(2,1)
                loss = loss_fn(y_pred, labels_segment)
                eval_loss += loss.item()

        print ("VALIDATION LOSS", (eval_loss / segments))