import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
train = pickle.load(f)
f.close()
n2 = len(train[0])
n1 = len(train)
f = open("N2ij_train.pkl", 'wb')
N = []

for i in range(n1):
    print(i)
    N_i = []
    for j in range(n1):
        N_ij = []
        for k in range(n2):
            if(train[i, k] and train[j, k]):  N_ij.append(k)
        N_i.append(N_ij)
    N.append(N_i)

pickle.dump(N, f)
f.close()