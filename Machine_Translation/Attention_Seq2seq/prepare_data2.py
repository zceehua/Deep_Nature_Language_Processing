# _*_coding:utf-8_*_
import time, threading
from multiprocessing import Pool,Process,RLock
import pickle
from collections import  Counter
from nltk.stem.wordnet import WordNetLemmatizer

import re
import math
import numpy as np
import os

specials = ['<GO>',  '<EOS>', '<PAD>', '<UNK>']
word2idx={}
idx2word={}
min_apperance=3
counter=Counter()
lemmatizer =WordNetLemmatizer()

for idx,value in enumerate(specials):
    word2idx[value]=idx
    idx2word[idx]=value
#
# print(word2idx)
# print(idx2word)
# decoder_input=[]
# encoder_input=[]

lemmatize=lambda x:lemmatizer.lemmatize(x,'v')

def preprocess(string):
    """Tokenization/string cleaning for a datasets.
    """
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", "  ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", "  ", string)
    string = re.sub(r"\"", " \" ", string)  # "a"       --> " a "
    string = re.sub(r"\'", "  ", string)  # they'     --> they '
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\-", "  ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", " ( ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", " ) ", string)  # ( they)   --> ( they )
    string = re.sub(r"\]", " ] ", string)  # they]     --> they ]
    string = re.sub(r"\[", " [ ", string)  # they[     --> they [
    string = re.sub(r"\?", " ? ", string)  # they?     --> they ?
    string = re.sub(r"\>", " > ", string)  # they>     --> they >
    string = re.sub(r"\<", " < ", string)  # they<     --> they <
    string = re.sub(r"\=", " = ", string)  # easier=   --> easier =
    string = re.sub(r"\;", " ; ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", "  ", string)
    string = re.sub(r"\:", " : ", string)  # easier:   --> easier :
    string = re.sub(r"\"", " \" ", string)  # easier"   --> easier "
    string = re.sub(r"\$", " $ ", string)  # $380      --> $ 380
    string = re.sub(r"\_", " _ ", string)  # _100     --> _ 100
    string = re.sub(r"\’", "", string)  # they’     --> they ’
    return string

def save_index(file):
    name=file[-2:]
    w2i = open("word2idx_{}.pkl".format(name), 'wb')
    i2w = open("idx2word_{}.pkl".format(name), 'wb')
    count=open("count_{}.pkl".format(name), 'wb')
    pickle.dump(word2idx, w2i)
    pickle.dump(idx2word, i2w)
    pickle.dump(counter, count)
    w2i.close()
    i2w.close()
    count.close()

def Reader(file_name,pool_num,length):
    cnt=0
    doc_id=1
    block_size=700000
    fd = open(file_name, 'r',encoding="utf-8")
    id=len(specials)
    for i in range(length):
        if cnt%block_size==0:
            if file_name[-2:]=='en':
                f = open("./encoder_train/index_{}.".format(doc_id) + file_name[-2:], 'w', encoding="utf-8")
            else:
                f = open("./decoder_train/index_{}.".format(doc_id) + file_name[-2:], 'w', encoding="utf-8")
            # if cnt!=0:
            #     save_index()
            doc_id+=1
        line=fd.readline()
        id_list = []
        line=preprocess(string=line)
        if file_name == "./data/train.en":
            line = re.sub(r"\s+", " ", line)  # Akara is    handsome --> Akara is handsome
            line= line.strip().lower()  # lowercase
            words_list = line.split(" ")
            words_list=list(map(lemmatize,words_list))
            counter.update(words_list)
            #print(counter)
        else:#"./data/train.zh":
            line = re.sub(r"\s+", "", line)
            words_list=list(line)
            counter.update(words_list)
        for word in words_list:
            if word not in word2idx:
                word2idx[word]=id
                idx2word[id]=word
                id +=1
            id_list.append(str(word2idx[word]))
        f.write(" ".join(id_list)+"\n")
        print(pool_num, "  ", cnt)
        cnt+=1
    save_index(file_name)
    f.close()
    fd.close()



if __name__ == '__main__':
    file_name = ["./data/train.en","./data/train.zh"]
    g = open(file_name[0], 'r', encoding="utf-8")
    length=len(g.readlines())
    print(length)
    if not os.path.exists("./encoder_train"):
        os.mkdir("./encoder_train")
    if not os.path.exists("./decoder_train"):
        os.mkdir("./decoder_train")
    num=2
    p=Pool(num)
    t=[]
    for i in range(num):
        p = Process(target=Reader,args=[file_name[i],i,length])
        t.append(p)
    for i in range(num):
        t[i].start()
    for i in range(num):
        t[i].join()

