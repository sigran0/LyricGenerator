
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random

from seq2seq_loader import Corpus, Dictionary
from seq2seq_model import BaseConfig, Encoder, Decoder
from dataset import Corpus_DataSet

use_cuda = torch.cuda.is_available()
print(use_cuda)
cuda_core = 0

batch_size = 50
epochs = 240

def variable_from_pair(pair):
    input_var = Variable(pair[0])
    target_var = Variable(pair[1])

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

    start_word_index = corpus.dictionary.get_index('사랑')

    decoder_input = Variable(torch.LongTensor(start_word_index))
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


# def evaluate(input_var, target_var, encoder, decoder, criterion, corpus):
#     loss = 0
#
#     encoder.eval()
#     decoder.eval()
#
#     target_len = target_var.size(0)
#     encoder_output, encoder_hidden = encoder(input_var)
#
#     decoder_input = Variable(torch.LongTensor([0]))
#     decoder_hidden = encoder_hidden
#
#     if use_cuda:
#         decoder_input = decoder_input.cuda(cuda_core)
#
#     for di in range(target_len):
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#         loss += criterion(decoder_output, target_var[di])
#         decoder_input = target_var[di]
#
#     return loss.data[0], target_len
#
#
# def evaluate_full(data):
#     total_loss = 0.0
#     total_len = 0
#
#     for pair in data:
#         input_var, target_var = variable_from_pair(pair)
#         cur_loss, cur_len = evaluate(input_var, target_var, encoder, decoder, criterion, corpus)
#         total_loss += cur_loss
#         total_len += cur_len
#     return total_loss / total_len


def run_train(dictionary):
    config = BaseConfig(len(dictionary))
    embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    encoder = Encoder(embedding=embedding, config=config)
    decoder = Decoder(embedding=embedding, config=config)

    if use_cuda:
        encoder.cuda(cuda_core)
        decoder.cuda(cuda_core)

    encoder_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = Corpus_DataSet(dictionary)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    for epoch in range(epochs):

        total_loss = 0.0

        for data in train_loader:
            # pair = list(zip(data, data))
            input_var, target_var = variable_from_pair(data)
            total_loss += train(input_var, target_var, encoder, decoder, encoder_optim, decoder_optim, criterion, dataset.corpus)

            print('Train : ', total_loss / 100)
            total_loss = 0.0

if __name__ == '__main__':
    path = 'data'

    dictionary = Dictionary(path)
    run_train(dictionary)


