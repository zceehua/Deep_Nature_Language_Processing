from data_utils import Dataloader
from config import args
import os
import pickle
from model import StructureModel
from sklearn.utils import shuffle

if not os.path.exists(args.save_path):
    print(args.save_path)
    files=[args.train_file,args.val_file,args.test_file]
    loader=Dataloader()
    print("data preprocessed !")


labels, docs, embedding, word2idx= pickle.load(open(args.save_path, 'rb'))
print("data loaded !")
train_X,train_y=docs["dev"],labels["dev"]
val_X,val_y=docs["dev"],labels["dev"]
#test_X,test_y=labels["test"],docs["test"]
vocab_size=len(word2idx)

train_X, train_y = shuffle(train_X, train_y, random_state=0)
val_X, val_y = shuffle(val_X, val_y, random_state=0)
print("data shuffled !")

model=StructureModel(embedding=embedding,vocab_size=vocab_size)
model.train(train_X,train_y,val_X,val_y)

