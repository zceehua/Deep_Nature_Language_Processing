import numpy as np

class Data(object):
    def __init__(self,context_file):
        self.context_file=context_file

    def read_user(self,file=None,num_u=5551,num_v=16980):
        f=open(file)
        R=np.mat(np.zeros((num_u,num_v)))
        for idx,line in enumerate(f):
            for value in line.strip().split()[1:]:
                R[idx,int(value)]=1
        return R

    def read_context(self):
        fp = open(self.context_file)
        lines = fp.readlines()
        X = []
        for i, line in enumerate(lines):
            words=[]
            strs = line.strip().split(' ')[1:]
            for strr in strs:
                segs = strr.split(':')
                words.append(int(segs[0]))
            X.append(words)

        return X

    def pad_input(self,X,max_len,padid):
        for i in range(len(X)):
            if len(X[i])<max_len:
                X[i].extend([padid]*(max_len-len(X[i])))
            else:
                X[i]=X[i][:max_len]
        return X
