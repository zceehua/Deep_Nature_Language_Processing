import numpy as np
import re

class Dataloader():
    def __init__(self,file,is_training=True,params=None,config=None):
        self.params={}
        self.config=config
        self.load_data(file)
        if is_training:
            self.build_vocab()
            self.to_index()
            self.pad(is_training)
        else:
            self.params=params
            self.to_index()
            self.pad(is_training)

    def load_data(self,file):
        self.sentences, self.tags = [], []
        sentence, tag = [], []
        f=open(file,'r',encoding="utf-8")
        for line in f.readlines():
            line=line.rstrip()
            if line:
                word=line.split()[0].lower() if self.config["lower"] else line.split()[0]
                word=re.sub('\d', '0', word) if self.config["zeros"] else word
                sentence.append(word)
                tag.append(line.split()[-1])
            else:
                if 'DOCSTART' not in sentence[0] and len(sentence)>0:
                    self.sentences.append(sentence)
                    self.tags.append(tag)
                    sentence,tag=[],[]
                else:
                    sentence,tag=[],[]
        for i in range(len(self.tags)):
            new_tags=iobes_iob(self.tags[i])
            self.tags[i]=new_tags


    def build_vocab(self):
        symbols = ['<pad>', '<unk>', '<start>', '<end>']
        words=list(set([w for sentence in self.sentences for w in sentence]))
        chars=list(set([char for word in words for char in word]))
        tags=list(set([tag for tags in self.tags for tag in tags]))
        self.params["word2idx"] ={word:idx for idx,word in enumerate(symbols+words)}
        self.params["idx2word"] ={idx:word for idx,word in enumerate(symbols+words)}
        self.params["char2idx"] ={char:idx for idx,char in enumerate(symbols+chars)}
        self.params["idx2char"]={idx:char for idx,char in enumerate(symbols+chars)}
        self.params["tag2idx"]={tag:idx for idx,tag in enumerate(symbols+tags)}
        self.params["idx2tag"]={idx:tag for idx,tag in enumerate(symbols+tags)}


    def to_index(self):
        self.sentences_wordidx,self.sentences_charidx,self.tagsidx=[],[],[]
        #self.sentence_len,self.char_len=[],[]
        for sentence in self.sentences:
            #self.sentence_len.append(len(sentence))
            self.sentences_wordidx.append([self.params["word2idx"].get(word,self.params["word2idx"]["<unk>"]) for word in sentence])
            charidx=[]
            #charlen=[]
            for word in sentence:
                charidx.append([self.params["char2idx"].get(char,self.params["char2idx"]["<unk>"]) for char in word])
                #charlen.append(len(word))
            self.sentences_charidx.append(charidx)
            #self.char_len.append(charlen)
        for tags in self.tags:
            self.tagsidx.append([self.params["tag2idx"].get(tag,self.params["tag2idx"]["<unk>"]) for tag in tags])

    def pad(self,is_training):
        if is_training:
            self.params["max_sent_len"]=max([len(s) for s in self.sentences])
            self.params["max_word_len"]=max([len(w) for s in self.sentences for w in s])


        for i in range(len(self.sentences_wordidx)):
            if self.params["max_sent_len"]>len(self.sentences_wordidx[i]):
                self.sentences_wordidx[i]+=[self.params["word2idx"]["<pad>"]]*(self.params["max_sent_len"]-len(self.sentences_wordidx[i]))
                self.tagsidx[i]+=[self.params["tag2idx"]["<pad>"]]*(self.params["max_sent_len"]-len(self.tagsidx[i]))
            else:
                self.sentences_wordidx[i]=self.sentences_wordidx[i][:self.params["max_sent_len"]]
                self.tagsidx[i]=self.tagsidx[i][:self.params["max_sent_len"]]



        for sentence in self.sentences_charidx:
            for i in range(len(sentence)):
                if self.params["max_word_len"]>len(sentence[i]):
                    sentence[i]+=[self.params["char2idx"]["<pad>"]]*(self.params["max_word_len"]-len(sentence[i]))
                else:
                    sentence[i]=sentence[i][:self.params["max_word_len"]]
            if self.params["max_sent_len"]>len(sentence):
                sentence+=[[self.params["char2idx"]["<pad>"]]*self.params["max_word_len"] for i in range(1,self.params["max_sent_len"]-len(sentence)+1)]
            else:
                sentence=sentence[:self.params["max_sent_len"]]



def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

