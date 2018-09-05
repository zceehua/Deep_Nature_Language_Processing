from data_utils import Corpus
from config import args
import os
import pickle
from model import MOS
from sklearn.utils import shuffle

if __name__ == '__main__':

    if args.nhidlast < 0:
        args.nhidlast = args.emsize
    if args.dropoutl < 0:
        args.dropoutl = args.dropouth


    data=Corpus(args.data)
    vocab_size=len(data.dictionary)

    train_data=data.train
    val_data=data.valid
    test_data=data.test

    model=MOS(vocab_size)
    model.train(train_data,val_data)