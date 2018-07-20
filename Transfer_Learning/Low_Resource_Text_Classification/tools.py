import logging
import os
import time
from config import args
import pickle
import tensorflow as tf
import  shutil
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from tqdm import tqdm
from collections import Counter

lemmatizer = lambda x:WordNetLemmatizer().lemmatize(x,"v")
stemmer = lambda x:PorterStemmer().stem(x)

def createLogger():
    logger = logging.getLogger("tensorflow")
    logger.setLevel(logging.DEBUG)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = args.log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)
    # logger.addHandler(console)
    return logger

def remove_model(path):
    shutil.rmtree(path)

def load_index(path):
    with open(path+"word2idx.pkl","rb") as f1:
        word2idx=pickle.load(f1)
    with open(path+"idx2word.pkl","rb") as f2:
        idx2word=pickle.load(f2)
    return word2idx,idx2word

def eval_confusion_matrix(labels, predictions,num_classes):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=4)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes,num_classes), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_op = tf.assign_add(con_matrix_sum, con_matrix)
        return tf.convert_to_tensor(con_matrix_sum), update_op


def load_embeddings(path, vocab, embed_dim, init=np.random.uniform):
    num_words = len((vocab))
    embedding_matrix = init(-0.15, 0.15, (num_words, embed_dim))
    f=open(path,'r',encoding='utf-8')
    for line in f.readlines():
        sp = line.split()
        sp[0]=stemmer(lemmatizer(sp[0]))
        if sp[0] in vocab:
            embedding_matrix[vocab[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype=np.float32)

    return embedding_matrix

def correct_spelling(string):
    string = re.sub(r"yrs", "years", string)
    string = re.sub(r"ouldnot", "should not", string)
    string = re.sub(r"paly", "play", string)
    return string

def clean_str(string,remove_stop=False):

    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", "", string)
    string = re.sub(r" s ", " ", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r" s ", " ", string)
    string = re.sub(r"A", " ", string)
    if remove_stop:
        string=" ".join([x for x in string.split() if x not in stopwords.words('english')])
    string=" ".join(list(map(lemmatizer,string.split())))
    string=" ".join(list(map(stemmer,string.split())))
    string=correct_spelling(string)
    return string.strip().lower()


def preprocess_dailymail(i,DATA_DIR,PARAMS):
    print("preprocessing and training on {}% of the daily mail files".format((i+1)*PARAMS['percent']*100))
    files=os.listdir(DATA_DIR)
    num_files=len(files)
    files=files[i*int(PARAMS['percent']*num_files):(i+1)*int(PARAMS['percent']*num_files)]
    text=""
    for file in tqdm(files, total=len(files), ncols=50):
        with open(os.path.join(DATA_DIR,file),encoding="utf-8") as f:
            text+=" "+f.read()
    text=clean_str(text,False)
    words = text.split()
    word2freq = Counter(words)
    words = [word for word in words if word2freq[word] > PARAMS['min_freq']]
    print("Total words:", len(words))

    if "word2idx" not in PARAMS:
        _words = list(set(words))
        PARAMS['word2idx'] = {c: i for i, c in enumerate(["<PAD>","<UNK>"]+_words)}
        PARAMS['idx2word'] = {i: c for i, c in enumerate(["<PAD>","<UNK>"]+_words)}
        PARAMS['vocab_size'] = len(PARAMS['idx2word'])
        print('Vocabulary size:', PARAMS['vocab_size'])
        with open("./embedding/dm_word2idx.pkl",'wb') as f1:
            pickle.dump(PARAMS['word2idx'],f1)
        with open("./embedding/dm_idx2word.pkl",'wb') as f2:
            pickle.dump(PARAMS['idx2word'],f2)

    indexed = [PARAMS['word2idx'].get(w,PARAMS['word2idx']["<UNK>"]) for w in words]
    #print("Word preprocessing completed ...")
    return indexed