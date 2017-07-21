import numpy as np
import os
import pickle
from metric import new_distance
import math
from find_Neighbour_New import find_Neighbour_New

dir = ".\\result"


def process_New_average_agg(num, K, beta, k, lam):
    d = dir + "\\" + str(num) + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", "rb")

    train_data = pickle.load(f)

    f.close()



    n1 = len(train_data)
    n2 = len(train_data[0])

    f = open(d + "item_beat_New.pkl", "rb")

    item_beat = pickle.load(f)

    f.close()


    N_u = find_Neighbour_New(num, beta, k)


    f = open(d + "true_rank.pkl", "rb")

    true_rank = pickle.load(f)

    f.close()

    f = open(d + "dis_uv_new.pkl", "rb")
    dis_uv = pickle.load(f)
    f.close()
    dis = np.zeros(n2)



    for u in range(n2):
        sigma = [0] * n1
        for i in range(n1):
            x = 0
            total = 0
            for v in N_u[u]:
                if(train_data[i, v] != -99):
                    x += math.exp(-lam * dis_uv[u][v]) * item_beat[i, v]
                    total += math.exp(-lam * dis_uv[u][v])
            if(total): sigma[i] = x/total
            #print(sigma[i])
        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        res = [x[1] for x in D]
        output_rank = {}
        for i in range(len(res)):
            output_rank[res[i]] = i + 1
            #if(i < K): output_rank[res[i]] = i + 1
            #else: output_rank[res[i]] = K + 1
        dis[u] = new_distance(true_rank[u], output_rank, K)
    return dis


    