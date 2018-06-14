
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random

from seq2seq_loader import Corpus
from seq2seq_model import BaseConfig, Encoder, Decoder

use_cuda = torch.cuda.is_available()
cuda_core = 0

def variable_from_pair(pair):
    input_var = Variable(torch.LongTensor(pair[0]))
    target_var = Variable(torch.LongTensor(pair[1]))

    if use_cuda:
        input_var = input_var.cuda(cuda_core)
        input_var = input_var.cuda(cuda_core)

    return input_var, target_var


def train(input_var, target_var, encoder, decoder, encoder_optim, decoder_optim, criterion, corpus):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss = 0

    encoder.train()
    decoder.train()

    target_len = target_var.size(0)
    encoder_output, encoder_hidden = encoder(input_var)

    decoder_input = Variable(torch.LongTensor([0]))
    decoder_hidden = encoder_hidden

    if use_cuda:
        decoder_input = decoder_input.cuda(cuda_core)

    for di in range(target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_var[di])
        decoder_input = target_var[di]

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)
    encoder_optim.step()
    decoder_optim.step()

    return loss.data[0] / target_len


def evaluate(input_var, target_var, encoder, decoder, criterion, corpus):
    loss = 0

    encoder.eval()
    decoder.eval()

    target_len = target_var.size(0)
    encoder_output, encoder_hidden = encoder(input_var)

    decoder_input = Variable(torch.LongTensor([0]))
    decoder_hidden = encoder_hidden

    if use_cuda:
        decoder_input = decoder_input.cuda(cuda_core)

    for di in range(target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_var[di])
        decoder_input = target_var[di]

    return loss.data[0], target_len


def evaluate_full(data):
    total_loss = 0.0
    total_len = 0

    for pair in data:
        input_var, target_var = variable_from_pair(pair)
        cur_loss, cur_len = evaluate(input_var, target_var, encoder, decoder, criterion, corpus)
        total_loss += cur_loss
        total_len += cur_len
    return total_loss / total_len

if __name__ == '__main__':
    corpus = Corpus('data')
    config = BaseConfig(len(corpus.dictionary))
    embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    encoder = Encoder(embedding=embedding, config=config)
    decoder = Decoder(embedding=embedding, config=config)

    if use_cuda:
        encoder.cuda(cuda_core)
        decoder.cuda(cuda_core)

    encoder_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    for c in range(1, 10000 + 1):
        idx = random.randint(0, len(corpus.train) - 1)
        input_var, target_var = variable_from_pair(corpus.train[idx])
        total_loss += train(input_var, target_var, encoder, decoder, encoder_optim, decoder_optim, criterion, corpus)

        if c % 100 == 0:
            print('Train : ', total_loss / 100)
            total_loss = 0.0
            print('Test : ', evaluate_full(corpus.test))
