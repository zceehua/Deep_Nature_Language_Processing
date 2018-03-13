import os
from config import args
from collections import Counter
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.batchify(self.tokenize(os.path.join(path, 'train.txt'))).astype(int)
        self.valid = self.batchify(self.tokenize(os.path.join(path, 'valid.txt'))).astype(int)
        self.test = self.batchify(self.tokenize(os.path.join(path, 'test.txt'))).astype(int)
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            #ids = torch.LongTensor(tokens)
            ids = np.zeros(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
    def batchify(self,data):
        n_batchs=len(data)//args.batch_size
        data=data[:n_batchs*args.batch_size]
        return data.reshape((args.batch_size,-1))

#data=Corpus(args.data)