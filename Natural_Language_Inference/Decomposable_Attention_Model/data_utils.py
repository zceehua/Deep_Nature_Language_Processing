import numpy as np
import nltk
from config import args
import pickle


class Dataloader():
    def __init__(self,files):
        self.data={}
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.load_embedding(args.embeddings)
        self.load_data(files)

    def load_data(self,files):
        for file in files:
            name=file.split("_")[-1][:-4]
            self.data.setdefault(name,{})
            f=open(file,encoding="utf-8")
            f.readline()
            pairs=[]
            while True:
                line=f.readline()
                if not line:
                    break
                if args.lower:
                    line=line.lower()
                line=line.strip().split("\t")
                label,sent1,sent2=line[0],line[5],line[6]
                if label=="-":
                    continue
                sent1=self.tokenizer.tokenize(sent1)
                #print(sent1)
                sent2=self.tokenizer.tokenize(sent2)
                pairs.append((label,sent1,sent2))

            labels=[pair[0] for pair in pairs]
            self.data["num_class"]=len(set(labels))
            labels_dict={label:idx for idx,label in enumerate(set(labels))}
            sent1=self.pad(self.convert2idx([pair[1] for pair in pairs]),name,"sent1_len",True)
            sent2=self.pad(self.convert2idx([pair[2] for pair in pairs]),name,"sent2_len",True)

            self.data[name]["sent1"]=sent1
            self.data[name]["sent2"]=sent2
            self.data[name]["label"]=np.array([labels_dict.get(label) for label in labels])

    def convert2idx(self,sents):
        return [[self.word2idx.get(word,self.word2idx["<UNK>"]) for word in sent] for sent in sents]

    def pad(self,sents,scope,name1,pad_go):
        self.data[scope][name1]=np.array([len(sent) for sent in sents])
        max_len=args.max_len
        if pad_go:
            self.data[scope][name1]+=1
            #max_len+=1
            sents=[[self.word2idx["<GO>"]]+sent for sent in sents]
        sents=[sent+[self.word2idx["<PAD>"]]*(max_len-len(sent)) for sent in sents ]
        return np.array(sents,dtype=np.int32)

    def load_embedding(self,file):
        f=open(file,encoding="utf-8")
        words=[]
        vectors=[]
        for line in f.readlines():
            line=line.strip()
            if line=="":
                continue
            line=line.split()
            words.append(line[0])
            vectors.append(np.array([float(x) for x in line[1:]],dtype=np.float32))
        vectors=np.array(vectors)

        words=["<PAD>","<GO>","<UNK>"]+words
        self.word2idx={word:idx for idx,word in enumerate(words)}
        self.data["word2idx"] = self.word2idx
        self.data["vocab_size"]=len(words)
        self.data["embedding"]=np.concatenate((np.random.uniform(-0.1, 0.1, (3,vectors.shape[1])),vectors))

    def save(self):
        pickle.dump(self.data, open(args.save_path, 'wb'))

# loader=Dataloader(["./data/snli_1.0_dev.txt"])
# print(loader.word2idx)
# print(len(loader.word2idx))
# print(loader.data["dev"]["sent1"].shape)
# print(loader.data["dev"]["sent2"].shape)
# print(loader.data["dev"]["label"].shape)
# print(loader.data["dev"]["sent1_len"].shape)
# print(loader.data["dev"]["sent2_len"].shape)
# print(loader.data["dev"]["sent2_len"])


