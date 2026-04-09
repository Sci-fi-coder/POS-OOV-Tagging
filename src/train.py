import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.dataset import create_batch


def train_model(model, dataset, vocab, epochs=3, batch_size=32):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.train()

    indices = list(range(len(dataset)))

    for epoch in range(epochs):

        total_loss = 0

        for i in tqdm(range(0, len(indices), batch_size)):

            batch_idx = indices[i:i+batch_size]

            words, chars, tags = create_batch(dataset, batch_idx)

            optimizer.zero_grad()

            outputs = model(words, chars)

            loss = criterion(
                outputs.view(-1, outputs.shape[-1]),
                tags.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")