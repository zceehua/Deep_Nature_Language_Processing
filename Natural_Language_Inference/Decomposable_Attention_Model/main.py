from data_utils import Dataloader
from config import args
import os
import pickle
from model import DecomposableNLIModel
from sklearn.utils import shuffle

if not os.path.exists(args.save_path):
    files=[args.train_file,args.val_file,args.test_file]
    loader=Dataloader(files)
    loader.save()
    data=loader.data
    print("data preprocessed !")
else:
    data= pickle.load(open(args.save_path, 'rb'))
    print("data loaded !")


data["train"]["sent1"],data["train"]["sent2"],data["train"]["label"] = \
    shuffle(data["train"]["sent1"],data["train"]["sent2"],data["train"]["label"], random_state=0)
data["dev"]["sent1"],data["dev"]["sent2"],data["dev"]["label"] = \
    shuffle(data["dev"]["sent1"],data["dev"]["sent2"],data["dev"]["label"], random_state=0)
print("data shuffled !")

model=DecomposableNLIModel(embedding=data["embedding"],num_class=data["num_class"],
                           embedding_size=data["embedding"].shape[1],vocab_size=data["vocab_size"])
model.train(data["train"],data["dev"])

