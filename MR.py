import numpy as np
import pickle
import math
import os
from metric import new_distance

dir = ".\\result\\"



def W(i, j, u, H, N_C):
    res = []
    for v in N_C[u]:
        if(H[i, v] != -99 and H[j, v] != -99): res.append(v)
    return res


def Pairwise_Rank(u, i, j, k, alg, H, R, N_C):
    #print(((u, i, j)))
    W_iju = W(i, j, u, H, N_C)
    W_iju.sort(key=lambda x: R[u][x], reverse= True)
    V = W_iju[0: k]

    if(V == []):
        if(alg[-8:] == "realvote"): return 0.5
        else: return np.random.randint(2)

    if(alg[-8:] == "realvote"):
        n = 0
        d = 0
        for v in V:
            if ((H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) == 1):
                if(alg == "MRW_realvote"): n += R[u][v]
                if(alg == "MR_realvote"): n += 1
            elif ((H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) == 0):
                if (alg == "MRW_realvote"): n += .5 * R[u][v]
                if (alg == "MR_realvote"): n += .5
            if (alg == "MRW_realvote"): d += R[u][v]
            if (alg == "MR_realvote"): d += 1

        return n / d

    P = []
    if(alg == "MR"):
        P = [(H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V]  # MR
    if(alg == "MRW"):
        P = [R[u][v] * (H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V] #MRW


    if(sum(P) > 0): return 1
    elif (sum(P) < 0): return 0
    return np.random.randint(2)



def Multi_Rank(k, H, true_rank, alg, R, N_C, K):

    n1 = len(H)
    n2 = len(H[0])
    dis = np.zeros(n2)

    for u in range(n2):
        #print(u)
        sigma = [0] * n1
        for j in range(n1):
            for i in range(j):
                if(H[i, u] != -99 and H[j, u] != -99):
                    # print(u, i, j)
                    sigma[i] += (H[i, u] > H[j ,u]) + np.random.randint(2) * (H[i, u] == H[j ,u])
                    sigma[j] += 1 - (H[i, u] > H[j ,u]) + np.random.randint(2) * (H[i, u] == H[j ,u])
                else:
                    #print((u, i, j))
                    tmp = Pairwise_Rank(u, i, j, k, alg, H, R, N_C)
                    sigma[i] += tmp
                    sigma[j] += 1 - tmp


        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        res = [x[1] for x in D]

        output_rank = {}
        for i in range(len(res)):
            output_rank[res[i]] = i + 1
        dis[u] = new_distance(true_rank[u], output_rank, K)

    return dis


def process_MR(num, alg, k, K): # alg is the version of MR algorithm, it may be MR, MRW, MR_realvote or MRW_realvote

    dir_t = dir + str(num) + "\\"

    if not os.path.exists(dir_t):
        os.makedirs(dir_t)

    f = open(dir_t + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    f = open(dir_t +"Ruv_train.pkl", 'rb')
    R = pickle.load(f)
    f.close()

    f = open(dir_t +"neighbour_users_beta.pkl", 'rb')
    N_C = pickle.load(f)
    f.close()

    f = open(dir_t +"true_rank.pkl", "rb")
    true_rank = pickle.load(f)
    f.close()

    dis = Multi_Rank(k, H, true_rank, alg, R, N_C, K)

    return dis




