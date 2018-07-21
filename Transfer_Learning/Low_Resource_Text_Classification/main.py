import os, sys
sys.path.append(os.getcwd())
from data.dataLoader import *
from data.recordLoader import *
import tensorflow as tf
from config import args
from tools import *
from model.model import Model
import numpy as np

#mysterious bugs might occur with tensorflow if not adding full path
FULL_PATH="D:/PycharmWorkSpace/Gits/Deep_Nature_Language_Processing/Transfer_Learning/Low_Resource_Text_Classification/model/saved_model/"
PRETRAIN_FULL_PATH="D:/PycharmWorkSpace/Gits/Deep_Nature_Language_Processing/Transfer_Learning/Low_Resource_Text_Classification/model/"
MODEL_PATH=args.model_path
RESULT_path=args.result_path


def result_writer(func):
    def inner_func(*args):
        eval_result, number_samples=func(*args)
        f.write('{accuracy:0.3f}'.format(**eval_result) + "\t\t{}\n".format(number_samples))
    return inner_func

@result_writer
def train(file_name,embedding):
    number_samples=file_name.split("_")[1][:-9]
    model = Model(logger, vocab_size,embedding=embedding)
    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn,model_dir="C:\\Users\\zt136\\AppData\Local\\Temp\\model_"+number_samples, config=config)
    for _ in range(args.n_epochs):
        classifier.train(lambda :recordLoader.train_input(file_name))
    logger.info("evaluating model on {} data samples....".format(number_samples))
    eval_result = classifier.evaluate(lambda :recordLoader.train_input("val.tfrecord"))
    logger.info('Val set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    del model,classifier
    # {'global_step': 500, 'accuracy': 0.571875, 'confusion_matrix': array([[65, 18, 7, 0],
    #                                                                       [30, 82, 23, 0],
    #                                                                       [8, 28, 36, 0],
    #                                                                       [0, 6, 17, 0]]), 'loss': 1.1649652}
    return eval_result,number_samples

def pre_train(input,embedding):
    model = Model(logger, vocab_size, embedding=embedding)
    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn, model_dir=PRETRAIN_FULL_PATH + "pretrain", config=config)
    for _ in range(args.n_epochs):
        classifier.train(lambda :lmloader.input_fn(input),steps=args.num_steps)

if __name__ == '__main__':
    logger = createLogger()
    loader = DataLoader(logger)
    config = tf.estimator.RunConfig(keep_checkpoint_max=1)
    if not os.path.exists("./data/processed_data"):
        logger.info("preprocessing data...")
        loader.load_data()

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    word2idx, idx2word = load_index(args.index_path)
    vocab_size = word2idx.__len__()

    logger.info("words index loaded,vocab size:{}".format(vocab_size))
    if args.allow_emb:
        embedding = load_embeddings("./embedding/glove.6B.200d.txt",
                                    word2idx, args.embedding_size, init=np.random.uniform)
        logger.info("word embedding loaded")
    else:
        embedding = None

    if not args.pretrain:
        recordLoader = RecordLoader()
        train_files = [x for x in os.listdir(args.save_path) if "train" in x]

        if not os.path.exists(RESULT_path):
            os.mkdir(RESULT_path)

        if os.path.exists(MODEL_PATH):
            remove_model(MODEL_PATH)

        with open(RESULT_path + "result_with_emb_with_pretrain.txt", 'w') as f:
            f.write("Val_acc\t\tNumber_of_training_data\n")
            for i in range(len(train_files)):
                args.load_model = True
                train(train_files[i],embedding)
    else:
        DATA_DIR = "F:/360Downloads/dailymail_stories/dailymail/stories/"
        PARAMS = {
            'min_freq': 5,
            'percent': 0.01,
            "word2idx":word2idx
        }
        lmloader=lmLoader()
        args.fine_tune = True
        #open domain pretrain
        logger.info("start open domain pretraining ...")
        args.amount=1
        for i in range(args.amount):
            index = preprocess_dailymail(i, DATA_DIR, PARAMS)
            pre_train(index,embedding)

        #in domain pretrain:
        logger.info("start in domain pretraining ...")
        args.fine_tune=True
        index=lmloader.load_gender_LM()
        pre_train(index, embedding)
        # index = lmloader.load_gender_LM()
        # print(index)
        # print(np.array(index).shape)