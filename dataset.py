
import torch.utils.data as data
from seq2seq_loader import Corpus

class Corpus_DataSet(data.Dataset):

    TYPE = ('train', 'validation', 'test')

    def __init__(self, dictionary, _type='train'):
        super(Corpus_DataSet, self).__init__()

        if _type not in self.TYPE:
            raise ValueError('Unknown split {:s}'.format(_type))

        self.corpus = Corpus('data', dictionary, _type)

    def __getitem__(self, index):
        item = self.corpus.target[index]
        idxs = self.corpus.words_to_idx(item)

        return idxs

    def __len__(self):
        return len(self.corpus.target)

