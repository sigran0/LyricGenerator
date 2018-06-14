
import torch.utils.data as data
from seq2seq_loader import corpus

class Corpus_DataSet(data.Dataset):

    TYPE = ('train', 'validation', 'test')

    def __init__(self, _type='train'):
        super(Corpus_DataSet, self).__init__()

        if type not in self.TYPE:
            raise ValueError('Unknown split {:s}'.format(_type))

        self.corpus = corpus('data')
