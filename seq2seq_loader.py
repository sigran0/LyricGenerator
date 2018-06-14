
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


class Corpus:

    TYPE = ('train', 'validation', 'test')

    def __init__(self, path, _type='train'):

        if _type not in self.TYPE:
            raise ValueError('Unknown type {:s}'.format(_type))

        self.dictionary = Dictionary(path)
        self._type = _type

        print(' > loading morphs data')

        self.train = self.tokenize(path, _type='train')
        self.test = self.tokenize(path, _type='test')
        self.validation = self.tokenize(path, _type='validation')

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

        data = []

        for lyric in lyric_all:
            lyric = self.words_to_idx(lyric)
            data.extend(list(zip(lyric[:, -1], lyric[1:])))

        data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)

        return data_train, data_test

    def words_to_idx(self, words):
        return list(map(lambda word: self.dictionary.get_index(word), words))

    def idx_to_words(self, idxs):
        return list(map(lambda idx: self.dictionary.get_word(idx), idxs))


corpus = Corpus('data')
test1 = corpus.train[0]

wix = corpus.words_to_idx(test1)
print(wix)
ixw = corpus.idx_to_words(wix)
print(ixw)