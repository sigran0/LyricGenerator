
import os
import pickle
from sklearn.model_selection import train_test_split


def load_pickle(filename, mode='rb'):
    with open(filename, mode=mode) as f:
        return pickle.load(f)


class Dictionary:

    def __init__(self, path):
        self.word2idx = load_pickle(path + '/wix.pkl')
        self.idx2word = load_pickle(path + '/ixw.pkl')

    def get_index(self, word):
        return self.word2idx[word]

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)


class Corpus:

    TYPE = ('train', 'validation', 'test')

    def __init__(self, path, dictionary, _type='train'):

        if _type not in self.TYPE:
            raise ValueError('Unknown type {:s}'.format(_type))

        self.dictionary = dictionary
        self._type = _type

        print(' > loading morphs data')

        self.target = self.tokenize(path, _type)

        print(' > loading morphs data complete')

    def tokenize(self, path, _type='train'):
        assert os.path.exists(path)

        path += '/' + _type

        lyric_all = []
        for filename in sorted(os.listdir(path)):
            try:
                morphs = load_pickle(path + '/' + filename)
                lyric_all.append(morphs)
            except Exception as e:
                print(e)
        print('   > type : {}, size : {}'.format(_type, len(lyric_all)))

        return lyric_all

    def words_to_idx(self, words):
        return list(map(lambda word: self.dictionary.get_index(word), words))

    def idx_to_words(self, idxs):
        return list(map(lambda idx: self.dictionary.get_word(idx), idxs))