
import torch.nn as nn
from torch.nn.functional import log_softmax


class BaseConfig:

    def __init__(self, vocab_size):
        self.rnn_type = 'LSTM'
        self.vocab_size = vocab_size
        self.embedding_size = 150
        self.hidden_size = 200
        self.n_layers = 2
        self.dropout = 0.2
        self.learning_rate = 1e-3


class Encoder(nn.Module):

    def __init__(self, embedding, config):
        super(Encoder, self).__init__()

        rnn_type = config.rnn_type
        embedding_size = config.embedding_size
        hidden_size = config.hidden_size
        n_layers = config.n_layers
        dropout = config.dropout

        self.embedding = embedding

        if rnn_type in ['LSTM', 'GRU', 'RNN']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, n_layers, dropout=dropout)
        else:
            raise ValueError('rnn_type must be LSTM or GRU or RNN')

    def forward(self, _input):
        embedded = self.embedding(_input).view(len(_input), 1, -1)
        output, hidden = self.rnn(embedded)
        return output, hidden


class Decoder(nn.Module):

    #   TODO    https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/05_RNN/2_Char_RNN/3_Char_RNN_LSTM.ipynb
    #   위에 참조해서 기초부터 다시 구현해보자

    def __init__(self, embedding, config):
        super(Decoder, self).__init__()

        rnn_type = config.rnn_type
        embedding_size = config.embedding_size
        hidden_size = config.hidden_size
        n_layers = config.n_layers
        dropout = config.dropout
        vocab_size = config.vocab_size

        self.embedding = embedding

        if rnn_type in ['LSTM', 'GRU', 'RNN']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, n_layers, dropout=dropout)
        else:
            raise ValueError('rnn_type must be LSTM or GRU or RNN')

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_word, last_hidden):
        embedded = self.embedding(input_word).view(1, 1, -1)


        output, hidden = self.rnn(embedded, last_hidden)

        output = output.squeeze(0)
        output = self.out(output)
        output = log_softmax(output, dim=1)

        return output, hidden