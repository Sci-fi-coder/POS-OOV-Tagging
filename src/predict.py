import torch
from src.dataset import pad_sequence

def predict_sentence(model, sentence, vocab):

    words = sentence.split()

    word_ids = []
    char_ids = []

    for word in words:

        w = word.lower()

        if w in vocab.word2idx:
            word_ids.append(vocab.word2idx[w])
        else:
            word_ids.append(vocab.word2idx["<UNK>"])

        chars = []

        for c in word:
            if c in vocab.char2idx:
                chars.append(vocab.char2idx[c])
            else:
                chars.append(vocab.char2idx["<UNK>"])

        char_ids.append(chars)

    max_char = max(len(c) for c in char_ids)

    char_ids = [pad_sequence(c, max_char) for c in char_ids]

    word_tensor = torch.tensor([word_ids])
    char_tensor = torch.tensor([[c for c in char_ids]])

    model.eval()

    with torch.no_grad():

        outputs = model(word_tensor, char_tensor)

        preds = torch.argmax(outputs, dim=-1)

    print("\nPrediction:\n")

    for word, tag_id in zip(words, preds[0]):

        tag = vocab.idx2tag[tag_id.item()]

        print(f"{word:15} -> {tag}") 