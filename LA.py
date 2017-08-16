import numpy as np
import os
import pickle
from metric import new_distance
import math
from utils import find_B_beta_LA

dir = ".\\result"


def W(u, i, v, j, lam, s_uv, s_ij):



    return math.exp(-lam * min(s_uv[u][v], s_ij[i][j]))




def process_LA(di, K, lam):
    d = dir + "\\" + di + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", "rb")

    train_data = pickle.load(f)

    f.close()
    '''
    f = open(d + "B_LA.pkl", "rb")
    B = pickle.load(f)
    f.close()
    '''

    n1 = len(train_data)
    n2 = len(train_data[0])

    f = open(d + "s_uv_LA.pkl", "rb")

    s_uv = pickle.load(f)

    f.close()

    f = open(d + "s_ij_LA.pkl", "rb")

    s_ij = pickle.load(f)

    f.close()

    s_ui, s_iu = find_B_beta_LA(di)

    f = open(d + "true_rank.pkl", "rb")


    true_rank = pickle.load(f)

    f.close()
    dis = np.zeros(n2)
    for u in range(n2):
        sigma = [0] * n1
        for i in range(n1):
            if(train_data[i, u] != -99): sigma[i] = train_data[i, u]
            else:
                nu = 0
                de = 0
                for j in s_iu[u][i]:
                    for v in s_ui[i][u]:
                        #if(u == v or j == i): continue
                        if(train_data[j, v] == -99): continue
                        w = W(u, i, v, j, lam, s_uv, s_ij)
                        nu += w
                        de += w * (train_data[j, u] + train_data[i, v] - train_data[j, v])
                if(nu): sigma[i] = de/nu

        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)

        res = [x[1] for x in D]
        output_rank = {}
        for i in range(len(res)):
            output_rank[res[i]] = i + 1
        dis[u] = new_distance(true_rank[u], output_rank, K)
        #print(dis[u])
    return dis