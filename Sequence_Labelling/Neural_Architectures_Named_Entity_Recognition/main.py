from config import args
# import tensorflow as tf
import numpy as np
from data_utils import Dataloader
import os
from model import BILSTM_CRF

def get_param():
    params={}
    params["char_embed"]=args.char_embed
    params["word_embed"]=args.word_embed
    params["char_hidden_size"]=args.char_hidden_size
    params["word_hidden_size"]=args.word_hidden_size
    params["train_file"]=args.train_file
    params["val_file"]=args.val_file
    params["test_file"]=args.test_file
    params["pre_embed"]=args.pre_embed
    params["dropout_rate"]=args.dropout_rate
    params["batch_size"]=args.batch_size
    params["zeros"]=args.zeros
    params["lower"]=args.lower
    params["use_pre_embed"]=args.use_pre_embed
    params["crf"]=args.crf
    params["grad_clip"]=args.grad_clip
    params["moving_average_decay"]=args.moving_average_decay
    params["n_layers"]=args.n_layers
    params["num_epochs"]=args.num_epochs
    params["decay_factor"]=args.decay_factor
    params["decay_size"]=args.decay_size
    params["lr"]=args.lr
    params["max_sent_len"]=args.max_sent_len
    return params

def gen_embeddings(path, vocab, embed_dim, init=np.random.uniform):

    num_words = len((vocab))
    embedding_matrix = init(-0.1, 0.1, (num_words, embed_dim))
    f=open(path,'r',encoding='utf-8')
    for line in f.readlines():
        sp = line.split()
        if sp[0] in vocab:
            embedding_matrix[vocab[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype=np.float32)

    return embedding_matrix


def main():
    params=get_param()
    loader=Dataloader(params["train_file"],config=params)
    val_loader=Dataloader(params["val_file"],is_training=False,params=loader.params,config=params)
    test_loader=Dataloader(params["test_file"],is_training=False,params=loader.params,config=params)
    params["max_sent_len"]=loader.params["max_sent_len"]
    params["max_word_len"]=loader.params["max_word_len"]
    params["word2idx"]=loader.params["word2idx"]
    params["char2idx"]=loader.params["char2idx"]
    params["tag2idx"]=loader.params["tag2idx"]

    if params["use_pre_embed"]==1:
        if os.path.exists(params["pre_embed"]):
            embedding_matrix=gen_embeddings(params["pre_embed"],params["word2idx"],params["word_embed"])
        else:
            raise Exception("None file")
    else:
        embedding_matrix = np.random.uniform(-0.1, 0.1, (len(params["word2idx"]), params["word_embed"]))


    X_train,y_train=(loader.sentences_wordidx,loader.sentences_charidx),loader.tagsidx
    X_val,y_val=(val_loader.sentences_wordidx,val_loader.sentences_charidx),val_loader.tagsidx
    X_test,y_test=(test_loader.sentences_wordidx,test_loader.sentences_charidx),test_loader.tagsidx

    print("data loaded")
    model=BILSTM_CRF(params,embedding_matrix=embedding_matrix)
    model.train(X_train,y_train,X_val,y_val,params["batch_size"])

main()
