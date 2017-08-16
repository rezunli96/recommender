import numpy as np
import pickle
import os
from metric import new_distance
import random
import math


dir = ".\\result"

def cal_dis_New(di, K):
    d = dir + "\\" + di + "\\"
    #print(d)
    #print("Calculating distance in New Algorithm for subsample", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()

    f = open(d + "observed_rank_new.pkl", "rb")
    observed_rank = pickle.load(f)
    f.close()

    n2 = len(data[0])
    n1 = len(data)

    f = open(d + "dis_uv_new.pkl", 'wb')

    dis = np.zeros((n2, n2))

    for u in range(n2):
        # print(u)
        for v in range(n2):
            u_rank_obs = observed_rank[u]
            v_rank_obs = observed_rank[v]
            dis[u][v] = new_distance(u_rank_obs, v_rank_obs, K)

    pickle.dump(dis, f)
    f.close()
    #print("Finish Calculating")


def cal_freq(di):
    d = dir + "\\" + di + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()
    n1 = len(H)
    n2 = len(H[0])
    total = 0
    for i in range(n1):
        for j in range(n2):
            if(H[i, j] != -99): total += 1
    #print("calculating observed ratio for subsample",num, "is ",total/(n1 * n2))
    return total/(n1 * n2)


def cal_true_rank(di):

    #print("Calculating True Ranking for ",  num)
    d = dir + "\\" + di + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "sampled_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    n1 = len(H)
    n2 = len(H[0])

    true_rank = []

    # f = open("true_rank.txt", "w")

    for u in range(n2):
        res = H[:, u]

        res = list(zip(res, range(len(res))))  # now res = [(rate(u, 0), 0), (rate(u, 1), 1).......]
        res.sort(key=lambda x: x[0], reverse=True) # u is now sorted decreasingly with respected to rate
        # res = list(zip([x[1] for x in res], range(1, len(res) + 1)))  # add rank number for each item
        # res.sort(key=lambda x: x[0]) # rearrange item with respect to their index
        res = [x[1] for x in res]
        rank = {}
        for i in range(len(res)):
            rank[res[i]] = i + 1
        # print(res)
        true_rank.append(rank)

    # print(true_rank)
    f = open(d + "true_rank.pkl", "wb")
    pickle.dump(true_rank, f)
    f.close()
    #print("Finshed.")


def cal_var_uv(data, d):
    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    n1 = len(data)
    n2 = len(data[0])

    s_uv = np.zeros((n2, n2))

    for u in range(n2):
        for v in range(n2):
            if(u != v and len(N_uv[u][v]) >= 2):
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



def cal_var(di):
    d = dir + "\\" + di + "\\"
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
                if (data[i, v] != -99): s_ui.append(v)
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
                if (data[j, u] != -99): s_iu.append(j)
            s_u.append(s_iu)
        s.append(s_u)
    return s


def find_B_beta_LA(di):
    d = dir + "\\" + di + "\\"
    # print("Finding possible neighbourhood in new algorithm for subsample", num,"with", (beta,k))
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

    return s_ui, s_iu

    '''
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
    '''



def find_Common_Items(di):
    d = dir + "\\" + di + "\\"
    #print("Finding Commonly Rated Item in train set for subsample ", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n2 = len(train[0])
    n1 = len(train)
    f = open(d + "Nuv_train.pkl", 'wb')
    N = []

    for u in range(n2):
        #print(u)
        N_u = []
        for v in range(n2):
            N_uv = []
            for l in range(n1):
                if (train[l, u] != -99 and train[l, v] != -99):  N_uv.append(l)
            N_u.append(N_uv)
            #print(len(N_uv))
        N.append(N_u)
    #print("Finished.")
    pickle.dump(N, f)
    f.close()



def find_Common_Users(di):
    d = dir + "\\" + di + "\\"
    #print("Finding Commonly Rated Item in train set for subsample ", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n2 = len(train[0])
    n1 = len(train)
    f = open(d + "Nij_train.pkl", 'wb')
    N = []

    for i in range(n1):
        #print(u)
        N_i = []
        for j in range(n1):
            N_ij = []
            for u in range(n2):
                if (train[i, u] != -99 and train[j, u] != -99):  N_ij.append(u)
            N_i.append(N_ij)
            #print(len(N_uv))
        N.append(N_i)
    #print("Finished.")
    pickle.dump(N, f)
    f.close()


def find_Neighbour_Items_beta(di, beta):
    #print("Find Neighbour Candidate for:", num)
    d = dir + "\\" + di + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    f = open(d + "Nij_train.pkl", 'rb')
    N_ij = pickle.load(f)
    f.close()

    f = open(d + "neighbour_items_beta.pkl", 'wb')
    n2 = len(H[0])
    n1 = len(H)

    #print(n2)


    N_candidate = []


    for i in range(n1):
        #print(u)
        Ni_candidate = []
        for j in range(n1):
            if (i != j):
                if (len(N_ij[i][j]) >= beta):
                    Ni_candidate.append(j)
        #print(len(Nu_candidate))
        N_candidate.append(Ni_candidate)

    pickle.dump(N_candidate, f)
    f.close()
    #print("Finished.")


def find_Neighbour_New(di, beta, k):
    d = dir + "\\" + di + "\\"
    #print("Finding possible neighbourhood in new algorithm for subsample", num,"with", (beta,k))
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)

    f = open(d + "dis_uv_new.pkl", 'rb')
    dis = pickle.load(f)
    f.close()

    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    N = []

    for u in range(n2):
        dis_u = dis[u]
        dis_u = list(zip(dis_u, range(len(dis_u))))
        dis_u.sort(key=lambda x: x[0])
        res = [x[1] for x in dis_u]
        res = [v for v in res if len(N_uv[u][v]) >= beta][:k]
        N.append(res)

    return N
    #print("Finishing Calculating.")


def find_Neighbour_Users_beta(di, beta):
    #print("Find Neighbour Candidate for:", num)
    d = dir + "\\" + di + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    f = open(d + "neighbour_users_beta.pkl", 'wb')
    n2 = len(H[0])
    n1 = len(H)

    #print(n2)


    N_candidate = []


    for u in range(n2):
        #print(u)
        Nu_candidate = []
        for v in range(n2):
            if (u != v):
                if (len(N_uv[u][v]) >= beta):
                    Nu_candidate.append(v)
        #print(len(Nu_candidate))
        N_candidate.append(Nu_candidate)

    pickle.dump(N_candidate, f)
    f.close()
    #print("Finished.")


def find_observed_rank_New(di, freq):
    #print("Finding observed rank for subsample", num)
    d = dir + "\\" + di + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)


    observed_rank = []
    for u in range(n2):
        rate = data[:, u]
        rate = list(zip(rate, range(len(rate))))
        rate.sort(key=lambda x: x[0], reverse=True)
        rate = [x[1] for x in rate if x[0] != -99]
        res = {}
        for i in range(len(rate)):
            rank_candidate = math.floor(i/freq) + 1
            res[rate[i]] = rank_candidate
            #if(rank_candidate <= K): res[rate[i]] = rank_candidate
            #else: res[rate[i]] = K + 1
        observed_rank.append(res)


    f = open(d + "observed_rank_new.pkl", "wb")

    pickle.dump(observed_rank, f)
    #print("Finish Calculating.")
    f.close()


def I(u, v, N):
    N_uv = N[u][v]
    res = []
    for i in range(int(len(N_uv) / 2)):
        res.append((N_uv[2 * i], N_uv[2 * i + 1]))
    return res



def R(u, v, H, N):
    res = 0
    I_uv = I(u, v, N)
    if (len(I_uv) == 0): return 0
    for (s, t) in I_uv:
        res += ((H[s, u] - H[t, u]) * (H[s, v] - H[t, v]) >= 0)
    return res / (len(I_uv))


def find_R_MR(num):
    #print("Finding Ruv for ", num)
    d = dir + "\\" + str(num) + "\\"
    #print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "Nuv_train.pkl", 'rb')
    N = pickle.load(f)
    f.close()

    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    n2 = len(H[0])
    n1 = len(H)

    # print(N)

    R_mat = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(n2):
            R_mat[i][j] = R(i, j, H, N)
        #print(i)

    f = open(d + "Ruv_train.pkl", 'wb')
    pickle.dump(R_mat, f)
    f.close()
    #print(R_mat)


def item_beat_New(di):
    #print("Finding item beat ratio for subsample", num)
    d = dir + "\\" + di + "\\"
    #print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n1 = len(train)
    n2 = len(train[0])
    res = np.zeros((n1, n2))
    for u in range(n2):
        for i in range(n1):
            if(train[i, u] == -99): continue
            beat = 0
            total = 0
            for j in range(n1):
                if(train[j, u] != -99):
                    total += 1
                    if(train[i, u] > train[j, u]): beat += 1
            if(total > 1): res[i, u] = beat/(total - 1)

    f = open(d + "item_beat_New.pkl", "wb")
    pickle.dump(res, f)
    f.close()
    #print("Finished.")


def generate_data(di, full_data, prob, user_size):
    #print("Generating dataset for ",num)
    d = dir + "\\" + di + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)


    total_user_number = len(full_data[0])
    sample = random.sample(range(total_user_number), user_size)
    sampled_data = full_data[:, sample]
    n1 = len(sampled_data)
    n2 = len(sampled_data[0])

    train = np.zeros((n1, n2))
    train.fill(-99)
    #val = np.zeros((n1, n2))
    #test = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            rnd = random.uniform(0, 100)
            if (rnd <= prob * 100 and sampled_data[i, j] != -99):   # split the dataset 40% into the training set
                train[i, j] = sampled_data[i, j]
            '''
            elif (rnd <= 55):
                val[i, j] = full_data[i, j]
            else:
                test[i, j] = full_data[i, j]
            '''

    f0 = open(d + "sampled_data.pkl", "wb")
    f1 = open(d + "train_data.pkl", 'wb')
    #f2 = open(d + "val_data.pkl", "wb")
    #f3 = open(d + "test_data.pkl", "wb")


    pickle.dump(sampled_data ,f0)
    pickle.dump(train, f1)

    print(train.shape)
    f0.close()
    f1.close()
    #print("Generating Finished.")


def generate_all(test_num, prob_list, data_set, user_size):
    f = open(data_set + ".pkl", "rb")
    full_data = pickle.load(f)
    f.close()
    for prob in prob_list:
        for num in range(test_num):
            di = str(prob) + "\\" + str(num)
            generate_data(di, full_data, prob, user_size)
            cal_true_rank(di)
            find_Common_Items(di)
            find_Common_Users(di)


def preprocess_LA(di, beta):
    find_Neighbour_Users_beta(di, beta)
    find_Neighbour_Items_beta(di, beta)
    #find_B_beta_LA(num)
    cal_var(di)

def preprocess_MR(num, beta):
    find_Neighbour_Users_beta(num, beta)
    find_R_MR(num)

def preprocess_new(di, beta, k , K):
    freq = cal_freq(di)
    find_observed_rank_New(di, freq)
    cal_dis_New(di, K)
    find_Neighbour_New(di, beta, k)
    item_beat_New(di)