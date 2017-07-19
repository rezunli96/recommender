import numpy as np
import os
import pickle
from metric import new_distance, kendall_tau
from find_Neighbour_New import find_Neighbour_New

dir = ".\\result"


def process_New_medium_agg(num, K, beta, k):
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


    dis = np.zeros(n2)
    #ken = np.zeros(n2)

    for u in range(n2):
        sigma = [0] * n1
        for i in range(n1):
            candidate = []
            for v in N_u[u]:
                if(train_data[i, v] != -99):
                    candidate.append(item_beat[i, v])
            candidate.sort()
            l = len(candidate)
            #print(l)
            #print(l//2)
            if(l):
                if(l % 2 == 0): sigma[i] = .5 * (candidate[l//2] + candidate[l//2-1])
                else: sigma[i] = candidate[(l-1)//2]
        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)
        res = [x[1] for x in D]
        output_rank = {}
        for i in range(len(res)):
            output_rank[res[i]] = i + 1
            #if(i < K): output_rank[res[i]] = i + 1
            #else: output_rank[res[i]] = K + 1
        a = [output_rank[i] for i in range(100)]
        b = [true_rank[u][j] for j in range(100)]
        dis[u] = new_distance(true_rank[u], output_rank, K)
        #ken[u] = kendall_tau(a, b)


    #print(np.mean(ken))
    return dis