import numpy as np
import re
from config import args
import gensim
import pickle
import os

class Dataloader():
    def __init__(self,files):
        self.load_file(files)
        self.word2vec()
        self.add_embedding()
        self.build_data()
        self.save_data()

    def load_file(self,files):
        self.labels,self.docs={},{}
        contents=[]
        for file in files:
            name=file.split(".")[-1]
            self.labels[name]=[]
            self.docs[name]=[]
            for line in open(file,encoding="utf-8"):
                line=line.strip().split("<split1>")
                self.labels[name].append(line[0])
                contents.append(line[1])
            for content in contents:
                content=content.split("<split2>")
                sentences=[re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sent) for sent in content]
                sent_tokens=[sent.split() for sent in sentences]
                sent_tokens=[sent for sent in sent_tokens if len(sent)!=0]# remove empty sent
                self.docs[name].append(sent_tokens)
            self.docs[name]=[doc for doc in self.docs[name] if len(doc)!=0] #remove empty doc

    def word2vec(self):
        sentences = []
        for doc in self.docs['train']:
            sentences.extend(doc)
        if 'dev' in self.docs:
            for doc in self.docs['dev']:
                sentences.extend(doc)
        print(sentences[0])
        if args.skip_gram:
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=args.word_emb_size, window=5, min_count=5, workers=4,
                                                             sg=1)
        else:
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=args.word_emb_size, window=5, min_count=5, workers=4)
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run=True)  # trim by min_count & precalculate downsampling
        print(rtn)
        self.w2v_model.finalize_vocab()  # build tables & arrays
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        self.vocab = self.w2v_model.wv.vocab
        self.vocab_size=len(self.vocab)
        print('Vocab size: {}'.format(self.vocab_size))

    def add_embedding(self):
        self.embedding=np.zeros((self.vocab_size+1,args.word_emb_size))
        for word in self.vocab:
            self.embedding[self.vocab[word].index]=self.w2v_model[word]
        self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        self.vocab = {word: self.w2v_model.wv.vocab[word].index for word in self.w2v_model.wv.vocab}

    def build_data(self):
        for key in self.docs:
            num_changed = 0
            for i in range(len(self.docs[key])):
                doc_len=len(self.docs[key][i])
                max_sent_len=max([len(sent) for sent in self.docs[key][i] ])

                if doc_len>args.max_sents:
                    num_changed+=1
                    self.docs[key][i]=self.docs[key][i][:args.max_sents]

                for j in range(len(self.docs[key][i])):
                    tokens_len=len(self.docs[key][i][j])
                    if tokens_len > args.max_tokens:
                        num_changed += 1
                        self.docs[key][i][j]=self.docs[key][i][j][:args.max_tokens]

                    self.docs[key][i][j]=[self.vocab.get(word,self.vocab["UNK"]) for word in self.docs[key][i][j]]
            print("number filtered for {}:{}".format(key,num_changed))
    def save_data(self):
        pickle.dump((self.labels, self.docs, self.embedding, self.vocab), open(args.save_path, 'wb'))


#loader=Dataloader([args.val_file])