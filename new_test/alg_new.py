import numpy as np
import random
import pickle
import os

import numpy as np
import pickle
import math
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
from metric import distance1


dir = ".\\result"

test_num = 100

T = 20


K = 30



def alg(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "complete_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    f = open(d + "train_data.pkl", 'rb')
    train_data = pickle.load(f)
    f.close()
    f = open(d + "true_rank.pkl", "rb")
    true_rank = pickle.load(f)
    f.close()
    f = open(d + "item_beat.pkl", 'rb')
    item_beat = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)
    f = open(d + "N_u.pkl", 'rb')
    N_u = pickle.load(f)
    f.close()
    dis = np.zeros(n2)
    for u in range(n2):
        sigma = [0] * n1
        for i in range(n1):
            x = 0
            total = 0
            for v in N_u[u]:
                if(train_data[i, v]):
                    x += item_beat[i, v]
                    total += 1
            if(total): sigma[i] = x/total
        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        D = list(zip([x[1] for x in D], range(1, len(D) + 1)))
        D.sort(key=lambda x: x[0])
        res = [x[1] for x in D]
        dis[u] = distance1(true_rank[u], res, K)

    f = open(d + "result.pkl", 'wb')
    pickle.dump(dis, f)
    print(dis)
    f.close()



for i in range(test_num):
    alg(i)

