import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os

'''
This file calculate the true ranking of each user and store it into true_rank.pkl. true_rank.pkl is a n2 * n1 matrix,

with each row represents the ranking of all n1 items for this user. For example, true_rank[u] = [3, 2, 5, 9 ....], 

meaning for user u, item 0 ranks 3 highest, item 1 ranks 2 highest.......

'''






dir = ".\\result"

def cal_true(num):

    print("Calculating True Ranking for ",  num)
    d = dir + "\\" + str(num) + "\\"
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
        res = list(zip([x[1] for x in res], range(1, len(res) + 1)))  # add rank number for each item
        res.sort(key=lambda x: x[0]) # rearrange item with respect to their index
        res = [x[1] for x in res]
        # print(res)
        true_rank.append(res)

    # print(true_rank)
    f = open(d + "true_rank.pkl", "wb")
    pickle.dump(true_rank, f)
    f.close()







