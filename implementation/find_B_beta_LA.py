import numpy as np
import pickle
import os


'''

This file finds B^beta_{u, i} in LA algorithm

'''


dir = ".\\result"


def find_beta_ovlp_i(data, d):
    f = open(d + "neighbour_users_beta.pkl", 'rb')
    N_u = pickle.load(f)
    f.close()

    s = []
    n1 = len(data)
    n2 = len(data[0])
    for i in range(n1):
        s_i = []
        for u in range(n2):
            s_ui = []
            for v in N_u[u]:
                if(data[i, v] != -99): s_ui.append(v)
            s_i.append(s_ui)
        s.append(s_i)
    return s
    
def find_beta_ovlp_u(data, d):
    f = open(d + "neighbour_items_beta.pkl", 'rb')
    N_i = pickle.load(f)
    f.close()

    s = []
    n1 = len(data)
    n2 = len(data[0])
    for u in range(n2):
        s_u = []
        for i in range(n1):
            s_iu = []
            for j in N_i[i]:
                if(data[j, u] != -99): s_iu.append(j)
            s_u.append(s_iu)
        s.append(s_u)
    return s


def find_B_beta_LA(num):
    d = dir + "\\" + str(num) + "\\"
    #print("Finding possible neighbourhood in new algorithm for subsample", num,"with", (beta,k))
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)

    s_ui = find_beta_ovlp_i(data, d)
    s_iu = find_beta_ovlp_u(data, d)

    B = []

    for i in range(n1):
        B_i = []
        for u in range(n2):
            B_iu = []
            for j in s_iu[u][i]:
                for v in s_ui[i][u]:
                    if(data[j, v] != -99): B_iu.append((j, v))
            B_i.append(B_iu)
        B.append(B_i)


    f = open(d + "B_LA.pkl", "wb")
    pickle.dump(B, f)
    f.close()
            