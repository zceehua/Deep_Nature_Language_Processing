import pandas as pd
import numpy as np
import os


from config import args
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import pickle
from tools import *

DATA_PATH=args.data_path
SAVE_PATH=args.save_path
NUM_ANSWERS=10
OFFSET=2
names=[""]

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

class DataLoader(object):
    def __init__(self,logger):
        self.questions=None
        self.answers=None
        self.labels=None
        self.word2idx={}
        self.idx2word={}
        self.logger=logger
        #self.load_data()

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            raise Exception("Data path not found!")
        for file in os.listdir(DATA_PATH):
            label=int(file.split("_")[-1][0])
            if self.questions is None or self.answers is None:
                data=pd.read_csv(DATA_PATH+file,skiprows=7,header=None)
                data.fillna("",inplace=True)
                self.questions=np.array([clean_str(x) for x in data.values[0][OFFSET:]])
                self.answers=np.zeros((len(data)-1,NUM_ANSWERS))
                self.labels=np.zeros((len(data)-1,NUM_ANSWERS))
                self.answers=self.preprocess(data.loc[1:,OFFSET:])
                self.labels[data.loc[1:,OFFSET:]!=""]=label
            else:
                data = pd.read_csv(DATA_PATH+file, skiprows=7, header=None)
                data.fillna("", inplace=True)
                self.answers+=self.preprocess(data.loc[1:,OFFSET:])
                self.labels[data.loc[1:,OFFSET:]!=""]=label

        self.answers,self.labels=np.array(self.answers),np.array(self.labels,np.int64)
        self.merged,self.labels=self.merge_shuffle()
        self.train_val_test_split()
        self.save(np.concatenate((self.x_train,self.y_train),-1),args.save_path)

        self.x_train=self.to_idx(self.x_train,mode="train")
        self.x_val=self.to_idx(self.x_val,mode="val")
        self.x_test=self.to_idx(self.x_test,mode="test")

        self.train=self.check_null(np.concatenate((self.x_train,self.y_train),axis=1))
        self.val=self.check_null(np.concatenate((self.x_val,self.y_val),axis=1))[:300]
        self.test=self.check_null(np.concatenate((self.x_test,self.y_test),axis=1))
        #print(len(self.val))

        n_split=int(len(self.merged)//args.train_split)
        for i in range(1,n_split+1):
            numbs=i*args.train_split
            if i==n_split:
                self.save_tfrecords(self.train, "train_{}".format(len(self.train)))
            else:
                self.save_tfrecords(self.train[:numbs],"train_{}".format(numbs))
        self.save_tfrecords(self.val,"val")
        self.save_tfrecords(self.test,"test")

    def train_val_test_split(self):
        nums = len(self.merged)
        def split(data):
            threshold=int(nums * args.split)
            threshold-=threshold%10
            train=data[:threshold]
            val=data[threshold:int(threshold+ (nums-threshold)/2)]
            test=data[int(threshold+ (nums-threshold)/2):]
            return train,val,test
        self.x_train,self.x_val,self.x_test = split(self.merged)
        self.y_train,self.y_val,self.y_test = split(self.labels)

    def to_idx(self,data,mode="train"):
        shape=data.shape
        #print(data.ravel()[:10])
        if mode=="train":
            symbols=["<PAD>","<UNK>"]
            words=[w  for sent in  data.ravel() for w in sent.split()]
            words.extend([x for x in self.questions])
            words=list(set(words))
            self.word2idx={word:idx for idx,word in enumerate(symbols+words)}
            self.idx2word={idx:word for idx,word in enumerate(symbols+words)}
            index=np.array([[self.word2idx.get(w,self.word2idx["<UNK>"]) for w in sent.split()] for sent in data.ravel()]).reshape(shape)
        else:
            index=np.array([[self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sent.split()] for sent in data.ravel()]).reshape(shape)
        return index

    def merge_shuffle(self):
        questions=np.tile(self.questions,[self.answers.shape[0],1]).reshape([-1,1])
        answers=np.reshape(self.answers,[-1,1])
        self.labels=np.reshape(self.labels,[-1,1])
        merged=np.concatenate((answers,questions),axis=1)
        del questions,answers
        return shuffle(merged,self.labels,random_state=10)


    def save(self,data,path):
        data=pd.DataFrame(data)
        if not os.path.exists(path):
            os.mkdir(path)
        data.to_csv(os.path.join(path,"processed.csv"),index=False)

    def preprocess(self,df,offset=2):
        for i in range(len(df.columns)):
            df[i+offset]=df[i+offset].apply(clean_str)
        return df

    def check_null(self,data):
        new_data=[]
        for row in data:
            if row[0]!=[]:
                new_data.append(row)
        new_data=new_data[:len(new_data)-len(new_data)%10]
        return new_data

    def save_tfrecords(self,data,mode):
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        self.logger.info("\n save tf-records file for {}".format(mode))
        path=os.path.join(SAVE_PATH,mode+".tfrecord")
        with tf.python_io.TFRecordWriter(path) as writer:
            for row in tqdm(data,total=len(data),ncols=50):
                answer,question,label=row[0],row[1],row[2]
                # example = tf.train.SequenceExample(
                #     # context
                #     context=tf.train.Features(feature={
                #         "label": _int64_feature(value=[label])
                #     }),
                #     # feature_lists
                #     feature_lists=tf.train.FeatureLists(feature_list={
                #         "answer": tf.train.FeatureList(feature=_int64_feature(value=answer)),
                #         "question": tf.train.FeatureList(feature=_int64_feature(value=question)),
                #     })
                # )
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'answer': _int64_feature(value=answer),
                            'question': _int64_feature(value=question),
                            'label': _int64_feature(value=[label]),
                        }))
                serialized = example.SerializeToString()
                writer.write(serialized)

    def save_wordidx(self):
        self.logger.info("saving words index...")
        if not os.path.exists(args.index_path):
            os.mkdir(args.index_path)
        with open(os.path.join(args.index_path,"word2idx.pkl"),"wb") as f1:
            pickle.dump(self.word2idx,f1)
        with open(os.path.join(args.index_path,"idx2word.pkl"),"wb") as f2:
            pickle.dump(self.idx2word, f2)

# loader=DataLoader()
# loader.load_data()
# print(loader)
# print(loader.questions)
# print(loader.answers[:10])
# print(loader.answers.shape)
# print(loader.questions.shape)
# print(loader.labels.shape)
# print(loader.labels[:10])

class dmLoader(object):

    def gen_batch(self,index):
        window=args.batch_size*args.max_len
        for i in range(0,len(index)-window,25):
            yield np.array(index[i:i+window]).reshape((-1,args.max_len)),\
                  np.array(index[i+1:i+1+window]).reshape((-1,args.max_len))

    @staticmethod
    def input_fn(self,index):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.gen_batch(index),
            (tf.int32, tf.int32),
            (tf.TensorShape([None, None]),
             tf.TensorShape([None, None])))
        iterator = dataset.make_one_shot_iterator()
        input, label = iterator.get_next()
        return ({'answer': input, 'question': input}, label)