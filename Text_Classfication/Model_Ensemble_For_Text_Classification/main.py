import re
import os,sys
import  tensorflow as tf
from tools import Dataloader,SklearnHelper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import numpy as np
from config import *
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import xgboost as xgb
import pickle
from cnn import CNN
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

def prepare_data():

    path=["./data/dataset.csv","./data/data/txt"]
    data_loader=Dataloader(path,class_dict)


    content,labels,cnn_input=data_loader.feature,data_loader.label,data_loader.cnn_input
    print("data loaded ")

    f = open('./trained/word2idx.pkl', 'wb')
    pickle.dump(data_loader.word2idx, f)
    f.close()

    vect = TfidfVectorizer(stop_words='english',min_df=2)

    X = vect.fit_transform(list(map(lambda x: " ".join(x),content)))
    f=open('./trained/vectorizer.pkl', 'wb')
    pickle.dump(vect, f)
    f.close()
    Y = np.array(labels,dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    num_train=len(X_train)
    num_test=len(X_test)

    return data_loader,cnn_input,Y,X_train, X_test, y_train, y_test,num_train,num_test

#get out of fold training and testing features
def get_oof(clf,x_train,y_train,x_test,num_train,num_test,kf):
    oof_train=np.zeros((num_train,))
    oof_test=np.zeros((num_test,))
    #oof_test_fold=np.zeros((K_fold,num_test))
    acc=0
    for i,(train_idx,test_idx) in enumerate(kf.split(x_train)):
        xtr=x_train[train_idx]
        ytr=y_train[train_idx]
        xval=x_train[test_idx]
        yval=y_train[test_idx]

        clf.fit(xtr,ytr)
        oof_train[test_idx]=clf.predict(xval)
        acc+=accuracy_score(yval,clf.predict(xval))
        #oof_test_fold[i,:]=clf.predict(x_test)

    f=open("./trained/"+clf.name+"_stack.pkl",'wb')
    pickle.dump(clf.clf,f)
    f.close()
    clf.fit(x_train,y_train)
    oof_test[:]=clf.predict(x_test)
    #print("Accuracy for single model {}: {} ".format(model.name, acc/K_fold))
    #oof_test[:]=np.mean(oof_test_fold,axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def clean_str(string):
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

if __name__ == '__main__':
    class_dict = {"business": 0, "entertainment": 1, "politics": 2, "sports": 3, "technology": 4, "health": 5,
                  "science": 6, "education": 7}
    class_dict_rev={value:key for key,value in class_dict.items()}
    optimize = False
    SEED = 42
    K_fold = 5
    kf = KFold(n_splits=K_fold)

    #train or infer
    if args.mode=="train":
        #print(len(cnn_input))
        #print(cnn_input[0])
        data_loader, cnn_input, Y, X_train, X_test, y_train, y_test, num_train, num_test = prepare_data()

        rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params,name="rf")
        et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params,name="et")
        gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params,name="gb")
        lg = SklearnHelper(clf=LogisticRegression, seed=SEED,params=lg_params,name="lg")

        classfiers = [rf,et,gb, lg]
        train_stack, test_stack = [], []

        for model in classfiers:
            if optimize:
                # please config you params if you want to optimize any model,for exmaple:
                # from hyperopt import fmin, tpe, hp, partial
                # params = {"max_depth": hp.choice('max_depth', range(2, 10)),
                #           "min_samples_split": hp.choice('min_samples_split', range(2, 10)),
                #           "max_features": hp.choice('max_features', range(50, 100)),
                #           "min_samples_leaf": hp.choice('min_samples_leaf', range(1, 10))}
                # best_param = rf.optimize(X_train, y_train, X_test, y_test, params_grid=params, eval_func=mean_absolute_error, max_evals=10)
                # print("optimization done!")
                pass
            print("starting model :", model.name)
            oof_train, oof_test = get_oof(model, X_train, y_train, X_test,num_train, num_test,kf)
            train_stack.append(oof_train)
            test_stack.append(oof_test)

        #appending cnn result
        sess=tf.Session()
        textcnn=CNN(sess=sess,word2idx=data_loader.word2idx,seq_len=args.max_len,num_class=len(class_dict))
        cnn_train,cnn_test=textcnn.train(cnn_input,Y,batch_size=64,seed=42)

        train_stack.append(cnn_train)
        test_stack.append(cnn_test)
        ens_train = np.concatenate(train_stack, axis=1)
        ens_test = np.concatenate(test_stack, axis=1)

        gbm = xgb.XGBClassifier(
            # learning_rate = 0.02,
            n_estimators=300,
            max_depth=2,
            # min_child_weight= 2,
            # gamma=1,
            # gamma=0.9,
            # subsample=0.8,
            # colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=-1).fit(ens_train, y_train)

        predictions = gbm.predict(ens_test)
        acc = accuracy_score(y_test, predictions)
        print("\nAccuracy for {}: {} ".format("xgb ensemble", acc))

        f = open("./trained/XGB_ens.pkl", "wb")
        pickle.dump(gbm, f)
        f.close()
    else:
        word2idx = pickle.load(open('./trained/word2idx.pkl', 'rb'))
        sess = tf.Session()
        textcnn = CNN(sess=sess, word2idx=word2idx, seq_len=args.max_len, num_class=len(class_dict))
        textcnn.load_model("./trained/")
        print("Please Enter your text:")
        for line in sys.stdin:
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()
            line=" ".join([stemmer.stem(lemmatizer.lemmatize(x, 'v')) for x in clean_str(string=line).split()])
            loaded_xgb = pickle.load(open("./trained/XGB_ens.pkl", "rb"))
            loaded_vect = pickle.load(open("./trained/vectorizer.pkl", "rb"))
            test_vec=loaded_vect.transform([line])
            test=[]
            classfiers=["rf_stack.pkl","et_stack.pkl","gb_stack.pkl","lg_stack.pkl"]

            for clf in classfiers:
                loaded_clf=pickle.load(open("./trained/"+clf,'rb'))
                test.append(loaded_clf.predict(test_vec.toarray()))
            test.append(textcnn.predict(line))
            test=np.array(np.concatenate(test)).reshape((1,-1))
            print(test)
            #print(test.toarray())
            prediction = loaded_xgb.predict(test)
            print(class_dict_rev[prediction[0]])