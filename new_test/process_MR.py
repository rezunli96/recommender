import numpy as np
import pickle
import math
import os
from metric import distance1, kendall_tau

dir = ".\\result\\"

K = 10

def W(i, j, u, H, N_C):
    res = []
    for v in N_C[u]:
        if(H[i, v] != -99 and H[j, v] != -99): res.append(v)
    return res


def Pairwise_Rank(u, i, j, k, alg, H, R, N_C):
    #print(((u, i, j)))
    W_iju = W(i, j, u, H, N_C)
    W_iju.sort(key=lambda x: R[u][x], reverse= True)
    #print(len(W_iju))
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
            #d += R[u][v]
            d += 1
        # print(n/d)
        return n / d

    P = []
    if(alg == "MR"):
        P = [(H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V]  # MR
    if(alg == "MRW"):
        P = [R[u][v] * (H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V] #MRW

    #print(sum(P))
    if(sum(P) > 0): return 1
    elif (sum(P) < 0): return 0
    return np.random.randint(2)



def Multi_Rank(k, H, true_rank, alg, R, N_C):
    n1 = len(H)
    n2 = len(H[0])
    dis = np.zeros(n2)
    ken = np.zeros(n2)
    for u in range(n2):
        #print(u)
        sigma = [0] * n1
        for j in range(n1):
            for i in range(j):
                if(H[i, u] != -99 and H[j, u] != -99):
                    # print(u, i, j)
                    sigma[i] += (H[i, u] > H[j ,u]) + np.random.randint(2) * (H[i, u] == H[j ,u])
                    sigma[j] += 1 - sigma[i]
                else:
                    #print((u, i, j))
                    sigma[i] += Pairwise_Rank(u, i, j, k, alg, H, R, N_C)
                    sigma[j] += 1 - sigma[i]


        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        D = list(zip([x[1] for x in D], range(1, len(D) + 1)))
        D.sort(key=lambda x: x[0])
        res = [x[1] for x in D]
        dis[u] = distance1(true_rank[u], res, K)
        ken[u] = kendall_tau(true_rank[u], res)
        #print(true_rank[u])
        #print(res)
        #print("\n")
    print("kendall_tau for MR: ", np.mean(ken), (np.var(ken)))
    return dis





def process_MR(num, alg): # alg is the version of MR algorithm, it may be MR, MRW, MR_realvote or MRW_realvote

    dir_t = dir + str(num) + "\\"

    dir_res = dir_t + alg + "\\"


    if not os.path.exists(dir_res):
        os.makedirs(dir_res)


    f = open(dir_t + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()


    f = open(dir_t +"Ruv_train.pkl", 'rb')
    R = pickle.load(f)
    f.close()

    f = open(dir_t +"Nuv_candidate.pkl", 'rb')
    N_C = pickle.load(f)
    f.close()

    f = open(dir_t +"true_rank.pkl", "rb")
    true_rank = pickle.load(f)
    f.close()

    n2 = len(H[0])
    n1 = len(H)

    k = 20


    dis = Multi_Rank(k, H, true_rank, alg, R, N_C)
    f = open(dir_t + "result_"+alg + ".pkl", 'wb')
    pickle.dump(dis, f)
    f.close()


    return dis




