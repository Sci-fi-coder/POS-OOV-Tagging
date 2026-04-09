from collections import Counter


class Vocab:

    def __init__(self):

        self.word2idx = {"<PAD>":0, "<UNK>":1}
        self.char2idx = {"<PAD>":0, "<UNK>":1}
        self.tag2idx = {}

    def build(self, sentences, tags):

        word_counter = Counter()
        char_counter = Counter()

        # count words and characters
        for sent in sentences:
            for word in sent:
                word_counter[word.lower()] += 1
                for c in word:
                    char_counter[c] += 1

        # build word vocabulary
        for word in word_counter:
            self.word2idx[word] = len(self.word2idx)

        # build character vocabulary
        for c in char_counter:
            self.char2idx[c] = len(self.char2idx)

        # build POS tag vocabulary
        for tag_seq in tags:
            for tag in tag_seq:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = len(self.tag2idx)

        self.idx2tag = {v:k for k,v in self.tag2idx.items()}