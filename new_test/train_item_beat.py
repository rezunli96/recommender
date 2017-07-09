import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os

dir = ".\\result"

test_num = 100

def item_beat(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n1 = len(train)
    n2 = len(train[0])
    res = np.zeros((n1, n2))
    for u in range(n2):
        for i in range(n1):
            if(not train[i, u]): continue
            beat = 0
            total = 0
            for j in range(n1):
                if(train[j, u]): total += 1
                if(train[j, u] and train[i, u] > train[j, u]): beat += 1
            if(total): res[i, u] = beat/total

    f = open(d + "item_beat.pkl", "wb")
    pickle.dump(res, f)
    f.close()


for i in range(test_num):
    item_beat(i)
