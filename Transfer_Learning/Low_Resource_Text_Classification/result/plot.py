import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def load_result(file):
    acc = []
    nums = []
    with open(file, "r") as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            acc.append(float(line.strip().split("\t\t")[0]))
            nums.append(int(line.strip().split("\t\t")[1]))
    acc = np.array(acc)
    nums = np.array(nums)
    return acc,nums

def predict(nums,acc,pred_nums):
    regr = LinearRegression()
    regr.fit(nums.reshape((-1, 1)), acc)
    pred_acc = regr.predict(pred_nums.reshape((-1, 1)))
    return pred_acc,regr.coef_,regr.intercept_

def plot_prediction():
    acc_with_pre_emb,nums_with_pre_emb=load_result("result_with_emb_with_pretrain.txt")

    #acc_with_pre_emb,nums_with_pre_emb=load_result("result_with_pre_emb.txt")

    #plt.plot(nums_with_pre_emb,acc_with_pre_emb)
    # plt.xlabel("Number of training data")
    # plt.ylabel('Valiation Accuracy')
    # plt.legend(["no_pre_emd","with_pre_emb"])
    # plt.savefig("result.jpg")
    # plt.figure()
    pred_nums = np.array([7000, 8000, 9000, 10000])
    #pred_acc=predict(nums_no_pre_emb,acc_no_pre_emb,pred_nums)
    #plt.plot(np.concatenate((nums_no_pre_emb,pred_nums)), np.concatenate((acc_no_pre_emb,pred_acc)))
    pred_acc,coef, intercept= predict(nums_with_pre_emb, acc_with_pre_emb, pred_nums)
    #plt.plot(np.concatenate((nums_with_pre_emb, pred_nums)), np.concatenate((acc_with_pre_emb, pred_acc)))
    plt.scatter(nums_with_pre_emb, acc_with_pre_emb)
    nums_no_pre_emb = list(nums_with_pre_emb)
    nums_no_pre_emb.extend([7000, 8000, 9000])
    nums_no_pre_emb = np.array(nums_no_pre_emb, dtype=np.float32)
    plt.plot(nums_no_pre_emb, (intercept + coef* nums_no_pre_emb))
    plt.xlabel("Number of training data")
    plt.ylabel('Valiation Accuracy')
    #plt.legend(["no_pre_emd","with_pre_emb"])
    plt.savefig("prediction.jpg")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,name="confusion_matrix.jpg"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name)
    #plt.show()

plot_prediction()
# np.set_printoptions(precision=2)
# classnames=["Inequitable","Low_Equitable","Moderately_Equitable","Equitable"]
# confusion_matrix=np.array([[71 ,11  ,2  ,1],[12 ,87 ,25, 3],[0 ,18 ,36, 12],[0  ,3  ,7 ,12]])
# plot_confusion_matrix(confusion_matrix,classnames,normalize=True,
#                       title='Confusion matrix, with normalization',name="confusion_matrix_norm.jpg")
# plot_confusion_matrix(confusion_matrix,classnames,normalize=False,
#                       title='Confusion matrix, without normalization',name="confusion_matrix_freq.jpg")
