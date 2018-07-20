import os,sys
sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
from tools import *
from embedding.skip_model import model_fn
import pickle

tf.logging.set_verbosity(tf.logging.DEBUG)
DATA_DIR="F:\\360Downloads\\dailymail_stories\\dailymail\\stories\\"
#DATA_DIR="./embedding\\dailymail\\stories\\"


PARAMS  =  {
    'min_freq': 5,
    'skip_window': 5,
    'n_sampled': 100,
    'embed_dim': 200,
    'batch_size': 500,
    'n_epochs': 1,
    'percent':0.1
}



def make_data(int_words):
    x, y = [], []
    for i in range(0, len(int_words)):
        input_w = int_words[i]
        labels = get_y(int_words, i)
        x.extend([input_w] * len(labels))
        y.extend(labels)
    return x, y


def get_y(words, idx):
    skip_window = np.random.randint(1, PARAMS['skip_window']+1)
    left = idx - skip_window if (idx - skip_window) > 0 else 0
    right = idx + skip_window
    y = words[left: idx] + words[idx+1: right+1]
    return list(set(y))

def save_embedding(embedding):
    with open("./embedding/word_embedding.pkl","wb") as f:
        pickle.dump(embedding,f)
    print("embedding saved")


remove_model("./embedding/saved/")
if os.path.exists("./embedding/dm_word2idx.pkl"):
    PARAMS["word2idx"], PARAMS["idx2word"] = load_index("./embedding/dm_")
for i in range(2):
    x_train, y_train = make_data(preprocess_dailymail(i,DATA_DIR,PARAMS))
    if "embedding" not in PARAMS:
        print("load embedding from glove.6B.200d.txt")
        PARAMS["embedding"]=load_embeddings("./embedding/glove.6B.200d.txt",
                                            PARAMS["word2idx"], PARAMS["embed_dim"], init=np.random.uniform)
    estimator = tf.estimator.Estimator(model_fn,params=PARAMS,model_dir="./embedding/saved")
    estimator.train(tf.estimator.inputs.numpy_input_fn(
        {"x":np.array(x_train)}, np.expand_dims(y_train, -1),
        batch_size = PARAMS['batch_size'],
        num_epochs = PARAMS['n_epochs'],
        shuffle = True))

embedding = estimator.evaluate(tf.estimator.inputs.numpy_input_fn(
    {"x":np.array(x_train)[0:100]}, np.expand_dims(y_train, -1)[0:100],shuffle = False))
embedding=np.array(list(embedding))
save_embedding(embedding)
