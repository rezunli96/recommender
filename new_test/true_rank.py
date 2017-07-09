import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os


dir = ".\\result"

test_num = 100


def true(num):

    print(num)
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "complete_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    n1 = len(H)
    n2 = len(H[0])

    true_rank = []

    # f = open("true_rank.txt", "w")

    for i in range(n2):
        u = H[:, i]

        u = list(zip(u, range(len(u))))
        u.sort(key=lambda x: x[0], reverse=True)
        # print(u)
        res = list(zip([x[1] for x in u if x[0]], range(1, len(u) + 1)))
        res.sort(key=lambda x: x[0])
        res = [x[1] for x in res]
        # print(res)
        true_rank.append(res)
        # f.write(",".join([str(x) for x in res])+'\n')

    # f.close()
    print(n2)

    # print(true_rank)
    f = open(d + "true_rank.pkl", "wb")
    pickle.dump(true_rank, f)
    f.close()



for i in range(test_num):
    true(i)







