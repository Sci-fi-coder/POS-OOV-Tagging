import torch


class POSDataset:

    def __init__(self, sentences, tags, vocab):
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def encode_word(self, word):

        word = word.lower()

        if word in self.vocab.word2idx:
            return self.vocab.word2idx[word]

        return self.vocab.word2idx["<UNK>"]

    def encode_chars(self, word):

        chars = []

        for c in word:

            if c in self.vocab.char2idx:
                chars.append(self.vocab.char2idx[c])
            else:
                chars.append(self.vocab.char2idx["<UNK>"])

        return chars

    def encode_tag(self, tag):
        return self.vocab.tag2idx[tag]

    def get_item(self, idx):

        words = self.sentences[idx]
        tags = self.tags[idx]

        word_ids = []
        char_ids = []
        tag_ids = []

        for word, tag in zip(words, tags):

            word_ids.append(self.encode_word(word))
            char_ids.append(self.encode_chars(word))
            tag_ids.append(self.encode_tag(tag))

        return word_ids, char_ids, tag_ids


def pad_sequence(seq, max_len, pad_value=0):

    return seq + [pad_value] * (max_len - len(seq))


def create_batch(dataset, batch_indices):

    batch_words = []
    batch_chars = []
    batch_tags = []

    # longest sentence in batch
    max_len = max(len(dataset.sentences[i]) for i in batch_indices)

    # longest word in batch (for char padding)
    max_char = 0
    for idx in batch_indices:
        words = dataset.sentences[idx]
        for w in words:
            max_char = max(max_char, len(w))

    for idx in batch_indices:

        word_ids, char_ids, tag_ids = dataset.get_item(idx)

        word_ids = pad_sequence(word_ids, max_len)
        tag_ids = pad_sequence(tag_ids, max_len, -1)

        batch_words.append(word_ids)
        batch_tags.append(tag_ids)

        padded_chars = []

        for c in char_ids:
            padded_chars.append(pad_sequence(c, max_char))

        # pad missing words
        padded_chars += [[0]*max_char]*(max_len-len(padded_chars))

        batch_chars.append(padded_chars)

    return (
        torch.tensor(batch_words),
        torch.tensor(batch_chars),
        torch.tensor(batch_tags)
    ) 