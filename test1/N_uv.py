import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
train = pickle.load(f)
f.close()
n2 = len(train[0])
n1 = len(train)
f = open("Nuv_train.pkl", 'wb')
N = []

for i in range(n2):
    print(i)
    N_i = []
    for j in range(n2):
        N_ij = []
        for k in range(n1):
            if(train[k, i] and train[k, j]):  N_ij.append(k)
        N_i.append(N_ij)
    N.append(N_i)

pickle.dump(N, f)
f.close()