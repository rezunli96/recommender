import numpy as np
import os
import pickle
from metric import distance2


dir = ".\\result"


def process_New(num, K):
    d = dir + "\\" + str(num) + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)


    f = open(d + "train_data.pkl", "rb")

    train_data = pickle.load(f)

    f.close()

    n1 = len(train_data)
    n2 = len(train_data[0])

    f = open(d + "item_beat.pkl", "rb")

    item_beat = pickle.load(f)

    f.close()

    f = open(d + "neighbour_in_new.pkl", "rb")

    N_u = pickle.load(f)

    f.close()

    f = open(d + "true_rank.pkl", "rb")

    true_rank = pickle.load(f)

    f.close()


    rank = []


    dis = np.zeros(n2)

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
        res = [x[1] for x in D]
        output_rank = {}
        for i in range(len(res)):
            if(i < K): output_rank[res[i]] = i + 1
            else: output_rank[res[i]] = K + 1
        dis[u] = distance2(true_rank[u], output_rank)

        #rank.append(res)


    return dis


    