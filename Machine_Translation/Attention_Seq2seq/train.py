import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import  os
import pickle
from collections import Counter
from model import NMT
from bs4 import BeautifulSoup
import argparse

block_size=700000
en_train="./encoder_train"
de_train="./decoder_train"
INDEX_PATH="./index/"

en_files=sorted(os.listdir(en_train))
de_files=sorted(os.listdir(de_train))


# def load_data(en_path=None,de_path=None):
#     en_input=[]
#     de_input=[]
#     f_en=open(en_path,'r',encoding='utf-8')
#     f_de=open(de_path,'r',encoding='utf-8')
#     for i in range(block_size):
#         en_input.append(f_en.readline().strip().split(" "))
#         de_input.append(f_de.readline().strip().split(" "))
#     return en_input,de_input

def load_idx():
    w2i_en = open(INDEX_PATH+"word2idx_en.pkl", 'rb')
    w2i_zh = open(INDEX_PATH+"word2idx_zh.pkl", 'rb')
    i2w_en = open(INDEX_PATH+"idx2word_en.pkl", 'rb')
    i2w_zh = open(INDEX_PATH+"idx2word_zh.pkl", 'rb')
    #counter_en=open("count_en.pkl",'rb')
    #counter_zh=open("count_zh.pkl",'rb')
    X_word2idx=pickle.load(w2i_en)
    Y_word2idx=pickle.load(w2i_zh)
    X_idx2word=pickle.load(i2w_en)
    Y_idx2word=pickle.load(i2w_zh)
    #c_en=pickle.load(counter_en)
    #c_zh=pickle.load(counter_zh)
    return X_word2idx,X_idx2word,Y_word2idx,Y_idx2word


def load_val(size=100):
    en_input = []
    de_input = []
    en_path = "./val_data_small/en_val.en"
    de_path = "./val_data_small/de_val.zh"
    f_en = open(en_path, 'r', encoding='utf-8')
    f_de = open(de_path, 'r', encoding='utf-8')
    for i in range(size):
        en_input.append(list(map(int, f_en.readline().strip().split(" "))))
        de_input.append(list(map(int, f_de.readline().strip().split(" "))))
    return  en_input,de_input

def load_val_large():
    f=open("valid.en-zh.en.sgm",'r',encoding='utf-8')
    content=f.read()
    soup=BeautifulSoup(content,"html.parser")
    content=soup.find_all('seg')
    en_input=[]
    for en in content:
        en_input.append(en.string)
    return  en_input
# def filter_low_freq(lowidx_en,lowidx_zh,X_word2idx,Y_word2idx,en_data,de_data):
#
#
#     for i in range(len(en_data)):
#         if en_data[i] in lowidx_en:
#             en_data[i]=X_word2idx['<UNK>']
#
#     for i in range(len(de_data)):
#         if de_data[i] in lowidx_zh:
#             de_data[i]=Y_word2idx['<UNK>']
#     return en_data,de_data

def train(epochs=10):

    num_splits=len(en_files)
    for epoch in range(1,epochs+1):
        file_id=1
        for en_file,de_file in zip(en_files,de_files):
            print("starting:",en_file,de_file)
            en_path=str(en_train)+"\\"+str(en_file)
            de_path=str(de_train)+"\\"+str(de_file)
            en_input = []
            de_input = []
            f_en = open(en_path, 'r', encoding='utf-8')
            f_de = open(de_path, 'r', encoding='utf-8')
            if en_file=="index_15.en":
                for i in range(103244) :
                    print(i)
                    en_input.append(list(map(int, f_en.readline().strip().split(" "))))
                    de_input.append(list(map(int, f_de.readline().strip().split(" "))))
            else:
                for i in range(block_size):
                    print("this is :",i)
                    en_input.append(list(map(int,f_en.readline().strip().split(" "))))
                    de_input.append(list(map(int,f_de.readline().strip().split(" "))))
            print("input_len:",len(en_input))
            model.train(en_input,de_input,en_val,de_val,epoch=epoch,file_id=file_id,file_nums=num_splits)
            file_id+=1

#model=main()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        default='train', help="train or infer ?")
    args = parser.parse_args()
    X_word2idx, X_idx2word, Y_word2idx, Y_idx2word = load_idx()
    en_val,de_val=load_val()
    model = NMT(X_word2idx=X_word2idx,X_idx2word=X_idx2word,Y_word2idx=Y_word2idx,Y_idx2word=Y_idx2word)
    if args.mode=="train": #python train_skip.py --mode train
        print("start training the model...")
        train()
    elif args.mode=="test":#python train_skip.py --mode test
        print("start inferring....")
        en_val=load_val_large()
        with open("translation_result",'w') as f:
            for en in en_val:
                output=model.infer(en)
                f.write(output+"\n")
            f.close()

