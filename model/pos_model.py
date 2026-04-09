import torch
import torch.nn as nn


class POSTagger(nn.Module):

    def __init__(self, vocab):

        super().__init__()

        word_vocab_size = len(vocab.word2idx)
        char_vocab_size = len(vocab.char2idx)
        tagset_size = len(vocab.tag2idx)

        word_dim = 100
        char_dim = 30
        char_hidden = 50
        lstm_hidden = 128

        # word embedding
        self.word_embed = nn.Embedding(word_vocab_size, word_dim)

        # character embedding
        self.char_embed = nn.Embedding(char_vocab_size, char_dim)

        # char LSTM
        self.char_lstm = nn.LSTM(
            char_dim,
            char_hidden,
            batch_first=True,
            bidirectional=True
        )

        # word BiLSTM
        self.word_lstm = nn.LSTM(
            word_dim + char_hidden*2,
            lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # classifier
        self.fc = nn.Linear(lstm_hidden*2, tagset_size)

    def forward(self, word_ids, char_ids):

        word_emb = self.word_embed(word_ids)

        batch, seq_len, char_len = char_ids.shape

        char_ids = char_ids.view(-1, char_len)

        char_emb = self.char_embed(char_ids)

        _, (h, _) = self.char_lstm(char_emb)

        char_rep = torch.cat((h[0], h[1]), dim=1)

        char_rep = char_rep.view(batch, seq_len, -1)

        combined = torch.cat((word_emb, char_rep), dim=2)

        lstm_out, _ = self.word_lstm(combined)

        logits = self.fc(lstm_out)

        return logits