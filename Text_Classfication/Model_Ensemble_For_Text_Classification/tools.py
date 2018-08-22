from scipy.sparse import csr_matrix, hstack
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from hyperopt import fmin,tpe,hp,partial
import re
from nltk.stem import PorterStemmer
from config import args

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None,name=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        self.name=name

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

    def optimize(self, x_train, y_train, x_val, y_val,params_grid,eval_func,max_evals):
        def loss(params):
            #global trial_counter
            self.clf=self.clf.set_params(**params)
            self.fit(x_train,y_train)
            result=eval_func(y_val,self.predict(x_val))
            #trial_counter+=1
            return result

        fn = lambda x: loss(x)
        algo = partial(tpe.suggest, n_startup_jobs=10)  # 1329
        best = fmin(fn, params_grid, algo=algo, max_evals=max_evals)
        self.clf.set_params(**best)
        return best


class Dataloader():
    def __init__(self,path,class_dict,max_len=args.max_len):
        self.max_len=max_len
        self.path=path
        self.class_dict=class_dict
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        self.lemmatize = lambda x: lemmatizer.lemmatize(x, 'v')
        self.stemmer=lambda x: stemmer.stem(x)
        self.load_csv(self.path[0])
        self.load_txt(self.path[1])
        self.build_vocab()
        self.build_cnn_input()

    def load_csv(self,csv_path):
        self.feature,self.label=[],[]
        data=pd.read_csv(csv_path,encoding="utf-8")
        news=data["news"].tolist()
        category=data["type"].tolist()
        for i in range(len(news)):
            #news[i]=" ".join(news[0].split())
            # print(news[i])
            # print(list(map(self.lemmatize, self.clean_str(news[i]).split())))
            lemmatized=list(map(self.lemmatize,self.clean_str(news[i]).split()))
            stemmed=list(map(self.stemmer,lemmatized))
            self.feature.append(stemmed)
            self.label.append(self.class_dict[category[i].lower()])

    def load_txt(self,txt_path):
        data = open("./data/data.txt", encoding="utf-8")
        pattern_n = "\['(.*?)'\]"
        pattern_l = "CATEGORY:(\w+)"

        for line in data.readlines():
            if len(line) != 1:
                line = re.sub(r"\[\"", "[\'", line)
                line = re.sub(r"\"\]", "\']", line)
                news = re.findall(pattern_n, line)
                category = re.findall(pattern_l, line)
                if news != ['']:
                    lemmatized = list(map(self.lemmatize, self.clean_str(news[0]).split()))
                    stemmed = list(map(self.stemmer, lemmatized))
                    self.feature.append(stemmed)
                    self.label.append(self.class_dict[category[0].lower()])

    def build_vocab(self):
        symbs = ['<pad>', '<unk>', '<start>', '<end>']
        vocab=list(set([word for content in self.feature for word in content ]))
        symbs.extend(vocab)
        self.word2idx={word:idx for idx,word in enumerate(symbs)}
        self.idx2word={idx:word for idx,word in enumerate(symbs)}

    def build_cnn_input(self):
        self.cnn_input=[]
        for content in self.feature:
            if len(content)>self.max_len:
                self.cnn_input.append([self.word2idx.get(word,self.word2idx['<unk>'])  for word in content[:self.max_len]])
            else:
                self.cnn_input.append([self.word2idx.get(word,self.word2idx['<unk>'])  for word in content]+[self.word2idx['<pad>']]*(self.max_len-len(content)))

    def clean_str(self,string):
        """
        Tokenization/string cleaning for datasets.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", "", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"'", "", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[0-9]\w+|[0-9]", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()