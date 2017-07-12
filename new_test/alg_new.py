import numpy as np
import random
import pickle
import os
import pickle
import math
import os
from metric import distance1, kendall_tau


dir = ".\\result"


K = 10


def new_alg(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "sampled_data.pkl", 'rb')
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
    precision = np.zeros(n2)
    rank = []
    ken = np.zeros(n2)



    for u in range(n2):
        sigma = [0] * n1
        for i in range(n1):
            x = 0
            total = 0
            for v in N_u[u]:
                if(train_data[i, v] != -99):
                    x += item_beat[i, v]
                    total += 1
            if(total): sigma[i] = x/total
        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        D = list(zip([x[1] for x in D], range(1, len(D) + 1)))
        D.sort(key=lambda x: x[0])
        res = [x[1] for x in D]
        rank.append(res)
        dis[u] = distance1(true_rank[u], res, K)
        ken[u] = kendall_tau(true_rank[u], res)
        # p = 0
    print("kendall_tau: ", np.mean(ken), (np.var(ken)))
    f = open(d + "dis_new.pkl", 'wb')
    pickle.dump(dis, f)
    f.close()
    f = open(d + 'res.pkl', "wb")
    pickle.dump(rank, f)
    f.close()
    return dis



