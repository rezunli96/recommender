import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os

dir = ".\\result"

test_num = 100

def N(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n2 = len(train[0])
    n1 = len(train)
    f = open(d + "Nuv_train.pkl", 'wb')
    N = []

    for u in range(n2):
        #print(u)
        N_u = []
        for v in range(n2):
            N_uv = []
            for l in range(n1):
                if (train[l, u] and train[l, v]):  N_uv.append(l)
            N_u.append(N_uv)
        N.append(N_u)

    pickle.dump(N, f)
    f.close()

for i in range(test_num):
    N(i)



