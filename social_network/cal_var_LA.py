import numpy as np
import pickle
import os


'''

This file compute variances s_uv^2 and s_ij^2 for each pair (u, v) and (i, j) in LA algorithm

'''


dir = ".\\result"


def cal_var_uv(data, d):
    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    n1 = len(data)
    n2 = len(data[0])

    s_uv = np.zeros((n2, n2))

    for u in range(n2):
        for v in range(n2):
            if(u != v):
                de = 0
                for i in range(len(N_uv[u][v])):
                    for j in range(i):
                        de += (data[N_uv[u][v][i], u] - data[N_uv[u][v][i], v] - (data[N_uv[u][v][j], u] - data[N_uv[u][v][j], v])) ** 2
                s_uv[u, v] = de/(2 * len(N_uv[u][v]) * (len(N_uv[u][v]) - 1))

    f = open(d + "s_uv_LA.pkl", "wb")
    pickle.dump(s_uv, f)
    f.close()


def cal_var_ij(data, d):
    f = open(d + "Nij_train.pkl", 'rb')
    N_ij = pickle.load(f)
    f.close()

    n1 = len(data)
    n2 = len(data[0])

    s_ij = np.zeros((n1, n1))

    for i in range(n1):
        for j in range(n1):
            if (i != j):
                if (len(N_ij[i][j]) >= 2):
                    de = 0
                    for u in range(len(N_ij[i][j])):
                        for v in range(u):
                            de += (data[i, N_ij[i][j][u]] - data[j, N_ij[i][j][u]] - (
                            data[i, N_ij[i][j][v]] - data[j, N_ij[i][j][v]])) ** 2
                    s_ij[i, j] = de / (2 * len(N_ij[i][j]) * (len(N_ij[i][j]) - 1))

    f = open(d + "s_ij_LA.pkl", "wb")
    pickle.dump(s_ij, f)
    f.close()



def cal_var(num):
    d = dir + "\\" + str(num) + "\\"
    #print("Finding Commonly Rated Item in train set for subsample ", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()

    n2 = len(data[0])
    n1 = len(data)

    cal_var_uv(data, d)
    cal_var_ij(data, d)



