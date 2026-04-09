import torch
from src.dataset import POSDataset, create_batch


def evaluate(model, sentences, tags, vocab):

    dataset = POSDataset(sentences, tags, vocab)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for i in range(len(dataset)):

            words, chars, gold = create_batch(dataset, [i])

            outputs = model(words, chars)

            preds = torch.argmax(outputs, dim=-1)

            mask = gold != -1

            correct += ((preds == gold) * mask).sum().item()
            total += mask.sum().item()

    accuracy = correct / total

    print("POS Tagging Accuracy:", accuracy)