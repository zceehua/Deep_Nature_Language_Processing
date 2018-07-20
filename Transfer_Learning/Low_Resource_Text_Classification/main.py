import os, sys
sys.path.append(os.getcwd())
from data.dataLoader import *
from data.recordLoader import *
import tensorflow as tf
from config import args
from tools import *
from model.model import Model
import numpy as np

#bugs with tens
#FULL_PATH="D:/PycharmWorkSpace/Gits/Deep_Nature_Language_Processing/Transfer_Learning/Low_Resource_Text_Classification"
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
        model_fn=model.model_fn, model_dir=MODEL_PATH+"model_"+number_samples, config=config)
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
        model_fn=model.model_fn, model_dir=MODEL_PATH + "model_pretrain", config=config)
    for _ in range(args.n_epochs):
        classifier.train(lambda :dmloader.input_fn(input))

if __name__ == '__main__':
    logger = createLogger()
    loader = DataLoader(logger)
    config = tf.estimator.RunConfig(keep_checkpoint_max=1)
    if not os.path.exists("./data/processed_data"):
        logger.info("preprocessing data...")
        loader.load_data()
    if not os.path.exists(args.index_path):
        loader.save_wordidx()
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    if not args.pretrain:
        word2idx, idx2word = load_index(args.index_path)
        logger.info("words index loaded")
        embedding = load_embeddings("./embedding/glove.6B.200d.txt",
                                    word2idx, args.embedding_size, init=np.random.uniform)
        logger.info("word embedding loaded")
        recordLoader = RecordLoader()
        vocab_size = word2idx.__len__()

        train_files = [x for x in os.listdir(args.save_path) if "train" in x]

        if not os.path.exists(RESULT_path):
            os.mkdir(RESULT_path)

        if os.path.exists(MODEL_PATH):
            remove_model(MODEL_PATH)
        with open(RESULT_path + "result_pre_emb.txt", 'w') as f:
            f.write("Val_acc\t\tNumber_of_training_data\n")
            for file_name in train_files:
                train(file_name,embedding)
    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        DATA_DIR = "F:/360Downloads/dailymail_stories/dailymail/stories/"
        PARAMS = {
            'min_freq': 5,
            'percent': 0.001
        }
        dmloader=dmLoader()

        if os.path.exists("./embedding/dm_word2idx.pkl"):
            PARAMS["word2idx"], PARAMS["idx2word"] = load_index("./embedding/dm_")
            vocab_size=len(PARAMS["word2idx"])
            embedding = load_embeddings("./embedding/glove.6B.200d.txt",
                                        PARAMS["word2idx"], args.embedding_size, init=np.random.uniform)
            logger.info("word embedding loaded")
        else:
            raise ValueError("please run train_skip.py to generate word2idx first")

        for i in range(2):
            index = preprocess_dailymail(i, DATA_DIR, PARAMS)
            # input,label=dmloader.input_fn(index)
            # sess.run(tf.global_variables_initializer())
            # x,y=sess.run([input,label])
            # print(x["answer"].shape)#(50,80)
            # print(y.shape)#(50,80)
            pre_train(index,embedding)
